# 🤖 Nexus Smart Assistant

Nexus is a modular, multimodal AI-powered personal assistant designed to combine **computer vision, face recognition, voice activity detection, speech-to-text, conversational AI, memory, and task-oriented tools** into a single intelligent assistant.

The project follows a modular architecture where individual AI capabilities are developed and evaluated independently and then integrated through a central orchestration layer.

---

## ✨ Features

### 🧠 Hybrid AI Brain

* Primary LLM: **Groq Cloud**
* Primary model: `llama-3.3-70b-versatile`
* Fallback LLM: **Ollama**
* Local fallback model: `phi3`
* Automatic fallback architecture for improved availability

### 👁️ Computer Vision

* Object/activity detection
* YOLO-based vision pipeline
* Model evaluation and inference scripts
* Training metrics and evaluation outputs

### 👤 Face Recognition

* Face detection and recognition
* Local face database
* Face embedding support
* Face recognition integration with the assistant

### 🎙️ Voice Activity Detection

* CNN-based Voice Activity Detection
* Audio classification
* Training and prediction pipeline
* Evaluation metrics and visualization

### 🗣️ Speech-to-Text

* Fine-tuned Whisper model integration
* Speech transcription
* Language testing
* Evaluation and prediction pipeline
* Local model support

> Large Whisper model weights are intentionally excluded from this repository because of their size.

### 💬 LLM Chatbot

* Natural-language interaction
* Groq-based cloud inference
* Ollama-based local fallback
* Tool integration
* Context-aware conversations

---

## 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │    Nexus Assistant  │
                         │  Main Integration   │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
       │     M1      │       │     M2      │       │     M3      │
       │ Computer    │       │    Face     │       │    Voice    │
       │   Vision    │       │ Recognition │       │   Activity  │
       └─────────────┘       └─────────────┘       │  Detection  │
                                                    └──────┬──────┘
                                                           │
                                                           ▼
                                                    ┌─────────────┐
                                                    │     M4      │
                                                    │   Whisper   │
                                                    │     STT     │
                                                    └──────┬──────┘
                                                           │
                                                           ▼
                                                    ┌─────────────┐
                                                    │     M5      │
                                                    │ Hybrid LLM  │
                                                    │    Brain    │
                                                    └──────┬──────┘
                                                           │
                              ┌────────────────────────────┼────────────────────┐
                              │                            │                    │
                              ▼                            ▼                    ▼
                       ┌─────────────┐              ┌─────────────┐      ┌─────────────┐
                       │   Memory    │              │    Tools    │      │  Responses  │
                       │  Manager    │              │  / Actions  │      │             │
                       └─────────────┘              └─────────────┘      └─────────────┘
```

---

## 📁 Project Structure

```text
Nexus/
│
├── m1/
│   ├── dataset.yaml
│   ├── train_m1.py
│   ├── evaluate_m1.py
│   └── infer_m1.py
│
├── m3_vad/
│   ├── train_vad.py
│   ├── plots/
│   └── results/
│
├── m4_stt/
│   ├── train/
│   ├── plots/
│   ├── results/
│   └── weights/
│
├── tests/
│   ├── check_yolo.py
│   ├── export_facenet_onnx.py
│   ├── test_vad_cnn.py
│   ├── test_whisper_lang.py
│   └── test_yolo_both.py
│
├── weights/
│
├── m2_face_recognition.py
├── m5_llm_chatbot.py
├── main_integration.py
├── memory_manager.py
├── tools.py
├── requirements.txt
├── .gitignore
├── .env.example
├── README.md
└── LICENSE
```

---

## 🛠️ Technology Stack

| Component                | Technology                         |
| ------------------------ | ---------------------------------- |
| Language                 | Python                             |
| LLM Cloud                | Groq                               |
| Cloud Model              | Llama 3.3 70B                      |
| Local LLM                | Ollama                             |
| Local Model              | Phi-3                              |
| Speech-to-Text           | OpenAI Whisper                     |
| Computer Vision          | YOLO                               |
| Face Recognition         | Face Recognition / Face Embeddings |
| Voice Activity Detection | CNN                                |
| Audio Processing         | Python audio processing libraries  |
| Model Evaluation         | Python / JSON / Markdown           |
| Environment              | Python Virtual Environment         |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/naruto-kavi/Nexus-Smart-Assistant.git
cd Nexus-Smart-Assistant
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 🔐 Environment Configuration

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit your real `.env` file to GitHub.

Use `.env.example` to document required variables without exposing secrets.

---

## 🤖 Local Ollama Setup

Nexus can use Ollama as a local fallback LLM.

Install Ollama and make sure the required model is available locally.

Example:

```powershell
ollama pull phi3
```

Then verify:

```powershell
ollama list
```

---

## ▶️ Running Nexus

From the project root:

```powershell
python main_integration.py
```

Alternatively:

```powershell
python -m main_integration
```

The main integration layer initializes the Nexus hybrid brain and connects the available modules.

---

## 🧪 Running Individual Modules

### M1 — Computer Vision

Train:

```powershell
python m1/train_m1.py
```

Evaluate:

```powershell
python m1/evaluate_m1.py
```

Inference:

```powershell
python m1/infer_m1.py
```

### M3 — Voice Activity Detection

```powershell
python m3_vad/train_vad.py
```

### M4 — Speech-to-Text

The M4 module contains the Whisper training/evaluation pipeline and generated evaluation results.

The large fine-tuned Whisper model is not included in GitHub because of its size.

---

## 🧠 Hybrid LLM Architecture

Nexus uses a two-level LLM strategy:

```text
                User Request
                     │
                     ▼
             ┌───────────────┐
             │ Nexus LLM     │
             │    Router     │
             └───────┬───────┘
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
        ┌──────────┐   ┌──────────┐
        │  Groq    │   │  Ollama  │
        │  Cloud   │   │  Local   │
        └──────────┘   └──────────┘
          Primary        Fallback
```

The primary path uses Groq for high-performance inference.

If the cloud model is unavailable or a local execution path is required, Nexus can use Ollama.

---

## 🔒 Security and Privacy

The repository intentionally excludes:

* API keys
* `.env`
* Python virtual environments
* Large model weights
* Runtime conversation logs
* Generated audio
* Local runtime databases/data
* Training artifacts that are not required for execution

Before publishing changes, verify:

```powershell
git status --ignored
```

---

## 📊 Model Evaluation

The repository contains evaluation outputs for several modules, including:

* Training curves
* Prediction results
* Evaluation metrics
* VAD metrics
* Whisper evaluation results

These artifacts provide visibility into model development and experimentation.

---

## 🧪 Testing

Available test scripts are located under:

```text
tests/
```

Examples include:

```powershell
python tests/test_vad_cnn.py
python tests/test_whisper_lang.py
python tests/test_yolo_both.py
```

---

## 👩‍💻 Author

**Bharkavi P M**

Nexus Smart Assistant is developed as a modular AI assistant project combining multiple machine-learning and generative-AI capabilities into a unified system.

---

## 📄 License

This project is licensed under the **MIT License**.

See [LICENSE](LICENSE) for details.
