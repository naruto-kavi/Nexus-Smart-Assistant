import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path

ROOT = Path(__file__).parent.parent
model_path = str(ROOT / "m4_stt" / "weights" / "whisper_best")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained(model_path)

audio_data = np.random.randn(16000).astype(np.float32)

input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
decoder_input_ids = torch.tensor([[processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]])

with torch.no_grad():
    outputs = model(input_features, decoder_input_ids=decoder_input_ids)
    
logits = outputs.logits[0, -1, :]

en_id = processor.tokenizer.convert_tokens_to_ids("<|en|>")
te_id = processor.tokenizer.convert_tokens_to_ids("<|te|>")

en_prob = logits[en_id].item()
te_prob = logits[te_id].item()

print(f"EN Prob (Logit): {en_prob}")
print(f"TE Prob (Logit): {te_prob}")

detected = "te" if te_prob > en_prob else "en"
print(f"Detected: {detected}")
