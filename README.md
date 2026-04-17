# Multimodal Deepfake Detection with Cross-Modal Attention and Explainable AI

> **Presented at the 6th International Conference on Recent Trends in Engineering, Technology and Management (ICRETEM 2026)**
> Suguna College of Engineering, Coimbatore, India — April 10–11, 2026

---

## 📌 Overview

This repository contains the full implementation of a multimodal deepfake detection framework that classifies media into four manipulation categories based on audio-visual signal analysis. Unlike binary real/fake detectors, this system identifies *which modality* has been tampered with, while also providing human-readable forensic explanations via Explainable AI (XAI) and LLM-generated reports.

**Average Cross-Dataset Accuracy: 86.6%** | **Macro-AUC on FakeAVCeleb: 0.980**

---

## 👥 Authors

| Name | Institution | Email |
|------|-------------|-------|
| Balasudhan C M | SRM IST, Ramapuram | bc0099@srmist.edu.in |
| Thilak Raaj N V | SRM IST, Ramapuram | nt9939@srmist.edu.in |
| Samuel Raj Irwin V | SRM IST, Ramapuram | si3076@srmist.edu.in |
| Sujatha K *(Advisor)* | SRM IST, Ramapuram | sujathak@srmist.edu.in |

---

## 🗂️ Repository Structure

```
├── FF_C23_Preprocess.ipynb        # Preprocessing for FaceForensics++ C23 dataset
├── LAV_DF_Preprocess.ipynb        # Preprocessing for LAV-DF dataset
├── FakeAVCeleb_Preprocess.ipynb   # Preprocessing for FakeAVCeleb dataset
├── Joint_Training.ipynb           # Joint multi-dataset model training
├── Inference.ipynb                # Inference, XAI visualization & forensic report generation
├── PROJECT_REPORT.docx            # Full project report
├── Multimodal_Deepfake_Detection_with_Cross-Modal_Attention_and_Explainable_AI.pdf  # Research paper
└── README.md
```

---

## 🧠 Method

### Four-Class Label Scheme

| Label | Video | Audio | Description |
|-------|-------|-------|-------------|
| 0 | Real | Real | Both modalities genuine |
| 1 | Fake | Real | Face swapped; audio untouched |
| 2 | Real | Fake | Voice cloned; video authentic |
| 3 | Fake | Fake | Both modalities synthetically generated |

### Datasets

| Dataset | Train | Test | Classes | Key Property |
|---------|-------|------|---------|--------------|
| FakeAVCeleb | ~17,600 | ~3,800 | 4 | Full AV manipulation coverage |
| LAV-DF | 800 | 320 | 4 | Temporally localised fakes (~9%) |
| FF++ C23 | 640 | 160 | 2 | Video-only; silent/absent audio |

### Model Architecture

The model is a multimodal fusion network with five principal components:

1. **Visual Encoder** — ResNet18 pretrained on ImageNet; produces 512-dim feature vector per frame
2. **Temporal Transformer with Frame Dropout** — 2-layer Transformer Encoder (d_model=512, 8 heads); frame dropout p=0.2
3. **Audio Encoder** — 3-layer CNN over 16 mel-spectrogram segments; outputs 256-dim audio embedding
4. **Cross-Modal Transformer** — Visual features (query) attend to audio features (key/value) via 8-head cross-attention
5. **AV Sync Feature** — Cosine similarity scalar between visual and audio embeddings; explicit desynchronization signal

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-4) |
| Loss Function | Focal Loss (γ=2.0) + Label Smoothing (0.1) |
| Epochs | 10 (early stopping) |
| Audio Dropout | p=0.3 |
| Frame Dropout | p=0.2 |
| Mixed Precision | AMP (fp16) |
| Gradient Clipping | Max norm 1.0 |

---

## 📊 Results

| Dataset | Accuracy | Macro-AUC | EER |
|---------|----------|-----------|-----|
| FakeAVCeleb | 90.4% | 0.980 | 0.9% |
| LAV-DF | 86.3% | 0.949 | 0.0% |
| FaceForensics++ C23 | 83.1% | 0.852 | 17.9% |
| **Average** | **86.6%** | — | — |

### Per-Class F1 Scores (FakeAVCeleb)

| Class | F1 Score |
|-------|----------|
| Real/Real | 92.9% |
| Fake Video / Real Audio | 89.3% |
| Real Video / Fake Audio | 89.6% |
| Fake/Fake | 91.2% |

---

## 🔍 Explainability (XAI)

Three complementary XAI techniques are used:

- **Grad-CAM** — Spatial activation maps over video frames highlighting regions most influential to the decision
- **Temporal Attention Weights** — Identifies the most informative frames within the sequence
- **Audio Saliency Maps** — Highlights frequency-time regions in mel-spectrograms relevant to the decision

These evidence streams are structured into a natural language prompt and passed to an **LLM**, which generates a human-readable forensic report — converting an opaque neural network decision into verifiable, auditable evidence suitable for forensic and legal contexts.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install librosa retinaface-pytorch
pip install numpy opencv-python ffmpeg-python
pip install matplotlib scikit-learn
```

### 1. Preprocess Datasets

Run each preprocessing notebook for the datasets you have downloaded:

```
FakeAVCeleb_Preprocess.ipynb
LAV_DF_Preprocess.ipynb
FF_C23_Preprocess.ipynb
```

Each notebook extracts 16 frames per video (224×224), generates 16 mel-spectrogram segments per audio track, and saves compressed `.npz` archives.

### 2. Train the Model

```
Joint_Training.ipynb
```

Trains the multimodal fusion model jointly across all three datasets using balanced sampling.

### 3. Run Inference & Generate Forensic Reports

```
Inference.ipynb
```

Runs the trained model on test samples, generates Grad-CAM visualizations, attention maps, audio saliency maps, and LLM-based forensic reports.

---

## 📋 Preprocessing Details

- **Face Detection**: RetinaFace — bounding box fixed from first 30 frames, applied to all subsequent frames
- **Frame Sampling**: 16 frames uniformly sampled per video; LAV-DF uses timestamp-aware sampling within annotated manipulation intervals
- **Audio**: Extracted at 16 kHz mono via FFmpeg; 16 mel-spectrogram segments (128 mel bins × 32 time steps) via librosa
- **Splits**: FakeAVCeleb — 80/20 by speaker identity; FF++ — 80/20 by manipulation type

---

## 📜 License

This project is released for academic and research purposes. Please contact the authors for any other use.
