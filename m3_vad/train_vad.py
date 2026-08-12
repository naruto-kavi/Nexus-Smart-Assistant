import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import json

# ==============================================================================
# Module 3: Voice Activity Detection (VAD) Training Script
# ==============================================================================

# --- Config ---
SR = 16000
CHUNK_DURATION = 0.5  # We'll use 500ms for training to get a (40, 32) spec
N_MELS = 40
TIME_STEPS = 32
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# --- Model Architecture (Module 3 Specs) ---
class VADCNN(nn.Module):
    def __init__(self):
        super(VADCNN, self).__init__()
        # Layer 1: 2D Conv (3x3) + ReLU + MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Layer 2: 2D Conv (3x3) + ReLU + MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Layer 3: 2D Conv (3x3) + ReLU
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Head: GlobalAvgPool -> Dense -> Sigmoid
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)

# --- Dataset ---
class VADDataset(Dataset):
    def __init__(self, speech_files, noise_files, transform=None):
        self.samples = []
        # Label 1 for speech, 0 for noise
        for f in speech_files:
            self.samples.append((f, 1))
        for f in noise_files:
            self.samples.append((f, 0))
        
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def extract_features(self, audio_path):
        try:
            # Load and fix duration
            y, sr = librosa.load(audio_path, sr=SR)
            target_len = int(SR * CHUNK_DURATION)
            
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
                
            # Extract Mel-spectrogram
            # To get 32 time steps from 500ms (8000 samples):
            # n_fft=512, hop_length=256 -> 8000/256 = 31.25 (approx 32)
            spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=N_MELS, n_fft=512, hop_length=256
            )
            spec = librosa.power_to_db(spec, ref=np.max)
            
            # Normalize to [0, 1]
            spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
            
            # Ensure shape (40, 32)
            if spec.shape[1] > TIME_STEPS:
                spec = spec[:, :TIME_STEPS]
            elif spec.shape[1] < TIME_STEPS:
                spec = np.pad(spec, ((0, 0), (0, TIME_STEPS - spec.shape[1])))
                
            return torch.FloatTensor(spec).unsqueeze(0) # (1, 40, 32)
        except Exception as e:
            return None

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        features = self.extract_features(path)
        
        # If extraction fails, pick a random sample (not ideal but robust for training)
        while features is None:
            idx = random.randint(0, len(self.samples)-1)
            path, label = self.samples[idx]
            features = self.extract_features(path)
            
        return features, torch.FloatTensor([label])

# --- Training Loop ---
def train():
    print("[*] Finding audio files...")
    speech_files = glob.glob(str(DATA_DIR / "LibriSpeech" / "train-clean-100" / "**" / "*.flac"), recursive=True)
    noise_files = glob.glob(str(DATA_DIR / "ESC-50-master" / "audio" / "*.wav"))
    
    if not speech_files or not noise_files:
        print(f"[ERROR] Datasets not found in {DATA_DIR}. Run download_vad_data.py first.")
        return

    print(f"    Speech samples: {len(speech_files)}")
    print(f"    Noise samples : {len(noise_files)}")

    # Limit samples for quicker training in this environment if needed
    # speech_files = speech_files[:2000]
    # noise_files = noise_files[:2000]

    dataset = VADDataset(speech_files, noise_files)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = VADCNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    print(f"[*] Starting training on {DEVICE}...")
    best_acc = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        # Store history
        history['train_loss'].append(train_loss/len(train_loader))
        history['val_loss'].append(val_loss/len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.4f} | Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_DIR / "vad_best.pt")
            print(f"    [SAVE] New best accuracy: {val_acc:.4f}")

    print("\n[*] Training Complete!")
    
    # Export to ONNX for real-time inference
    print("[*] Exporting to ONNX...")
    model.load_state_dict(torch.load(WEIGHTS_DIR / "vad_best.pt"))
    model.eval()
    dummy_input = torch.randn(1, 1, N_MELS, TIME_STEPS).to(DEVICE)
    torch.onnx.export(model, dummy_input, str(WEIGHTS_DIR / "vad_model.onnx"), 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"    ✓ ONNX model saved to {WEIGHTS_DIR / 'vad_model.onnx'}")

    # --- Analysis & Visualization ---
    print("\n[*] Generating Analysis Plots...")
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='s')
    plt.title('VAD Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', marker='o')
    plt.plot(history['val_acc'], label='Val Acc', marker='s')
    plt.title('VAD Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    analysis_path = ROOT / "training_analysis.png"
    plt.tight_layout()
    plt.savefig(str(analysis_path))
    print(f"    ✓ Analysis plot saved to {analysis_path}")

    # Show Prediction Examples
    print("\n[*] Random Prediction Examples (Validation Set):")
    model.eval()
    with torch.no_grad():
        indices = random.sample(range(len(val_ds)), 5)
        for idx in indices:
            feat, label = val_ds[idx]
            output = model(feat.unsqueeze(0).to(DEVICE))
            pred = "SPEECH" if output.item() > 0.5 else "NOISE"
            actual = "SPEECH" if label.item() == 1 else "NOISE"
            conf = output.item() if pred == "SPEECH" else 1.0 - output.item()
            print(f"    Sample {idx:04d}: Actual={actual:7s} | Pred={pred:7s} | Conf={conf:.2%}")

if __name__ == "__main__":
    train()
