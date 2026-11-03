# Nexus Smart Assistant

Nexus is a modular, multimodal AI-powered personal assistant designed to combine **computer vision, face recognition, voice activity detection, speech-to-text, conversational AI, memory, and task-oriented tools** into a single intelligent assistant.

The project follows a modular architecture where individual AI capabilities are developed and evaluated independently and then integrated through a central orchestration layer.

---

## Features

## Hybrid AI Brain

* Primary LLM: **Groq Cloud**
* Primary model: `llama-3.3-70b-versatile`
* Fallback LLM: **Ollama**
* Local fallback model: `phi3`
* Automatic fallback architecture for improved availability

## Computer Vision

* Object/activity detection
* YOLO-based vision pipeline
* Model evaluation and inference scripts
* Training metrics and evaluation outputs

## Face Recognition

* Face detection and recognition
* Local face database
* Face embedding support
* Face recognition integration with the assistant

## Voice Activity Detection

* CNN-based Voice Activity Detection
* Audio classification
* Training and prediction pipeline
* Evaluation metrics and visualization

## Speech-to-Text

* Fine-tuned Whisper model integration
* Speech transcription
* Language testing
* Evaluation and prediction pipeline
* Local model support

> Large Whisper model weights are intentionally excluded from this repository because of their size.

## LLM Chatbot

* Natural-language interaction
* Groq-based cloud inference
* Ollama-based local fallback
* Tool integration
* Context-aware conversations

---

## Architecture

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

## Hybrid LLM Architecture

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

## Model Evaluation

The repository contains evaluation outputs for several modules, including:

* Training curves
* Prediction results
* Evaluation metrics
* VAD metrics
* Whisper evaluation results

These artifacts provide visibility into model development and experimentation.

---

## Testing

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

## License

This project is licensed under the **MIT License**.

See [LICENSE](LICENSE) for details.
