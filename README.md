# Multimodal Deepfake Detection with Cross-Modal Attention and Explainable AI

> **Presented at the 6th International Conference on Recent Trends in Engineering, Technology and Management (ICRETEM 2026)**  
> Suguna College of Engineering, Coimbatore, India — April 10–11, 2026

---

## 📌 Overview

This repository contains the full implementation of a **multimodal deepfake detection framework** that classifies video media into four distinct manipulation categories by jointly analysing audio and visual signals. Unlike conventional binary real/fake detectors, this system identifies *which modality* has been tampered with — face-swapped video, voice-cloned audio, or both — making it significantly more informative for forensic investigations.

The framework additionally provides **human-readable forensic reports** via three complementary Explainable AI (XAI) techniques (Grad-CAM, temporal attention, audio saliency), all synthesised into natural-language prose by an **LLM (Gemini 2.5 Flash Lite)**. A fully self-contained **Web UI** deployed via Flask + ngrok allows anyone with a browser to upload a video and receive a complete forensic analysis in seconds.

| Metric | Value |
|--------|-------|
| Average Cross-Dataset Accuracy | **86.6%** |
| Macro-AUC on FakeAVCeleb | **0.980** |
| EER on LAV-DF | **0.0%** |

---

## 👥 Authors

| Name | Role | Institution | Email | GitHub |
|------|------|-------------|-------|--------|
| Balasudhan C M | Author | SRM IST, Ramapuram | bc0099@srmist.edu.in | [Balasudhan123](https://github.com/Balasudhan123) |
| Thilak Raaj N V | Author | SRM IST, Ramapuram | nt9939@srmist.edu.in | [TRJgit](https://github.com/TRJgit) |
| Samuel Raj Irwin V | Author | SRM IST, Ramapuram | si3076@srmist.edu.in | [Samuel004](https://github.com/Samuel004) |
| Sujatha K | Advisor | SRM IST, Ramapuram | sujathak@srmist.edu.in | — |

---

## 🗂️ Repository Structure

```
├── FF_C23_Preprocess.ipynb          # Preprocessing for FaceForensics++ C23 dataset
├── LAV_DF_Preprocess.ipynb          # Preprocessing for LAV-DF dataset
├── FakeAVCeleb_Preprocess.ipynb     # Preprocessing for FakeAVCeleb dataset
├── Joint_Training.ipynb             # Joint multi-dataset model training
├── Inference.ipynb                  # Inference, XAI visualisation & forensic report generation
├── Deepfake_WebUI_2_1_.ipynb        # Flask + ngrok web application for live inference
├── PROJECT_REPORT.docx              # Full project report
├── Multimodal_Deepfake_Detection_with_Cross-Modal_Attention_and_Explainable_AI.pdf
└── README.md
```

---

## 🧠 Problem Formulation

### Four-Class Label Scheme

Standard deepfake detectors output a binary real/fake score, which is insufficient for forensic purposes — an investigator needs to know *what* was manipulated. This system outputs one of four classes:

| Label | Video | Audio | Description |
|-------|-------|-------|-------------|
| 0 | ✅ Real | ✅ Real | Both modalities are genuine |
| 1 | 🎭 Fake | ✅ Real | Face swapped or reenacted; audio untouched |
| 2 | ✅ Real | 🎤 Fake | Voice cloned or synthesised; video authentic |
| 3 | 🎭 Fake | 🎤 Fake | Both modalities are synthetically generated |

---

## 🗃️ Datasets

Three publicly available datasets are used for training and evaluation, chosen to cover the full spectrum of manipulation types:

| Dataset | Total Clips | Train | Test | Classes | Key Property |
|---------|-------------|-------|------|---------|--------------|
| **FakeAVCeleb** | ~21,400 | ~17,600 | ~3,800 | 4 | Full audio-visual manipulation coverage; celebrity subjects |
| **LAV-DF** | 1,120 | 800 | 320 | 4 | Temporally *localised* fakes (~9% of frames); hardest test |
| **FF++ C23** | 800 | 640 | 160 | 2 | Video-only (silent/absent audio); C23 lossless compression |

**FakeAVCeleb** provides the primary benchmark as it covers all four manipulation classes. **LAV-DF** tests robustness under sparse, temporal manipulation. **FF++ C23** evaluates video-only performance with no audio signal.

---

## 🏗️ Model Architecture

The core model is a **multimodal fusion network** with five principal components assembled into an end-to-end differentiable pipeline.

```
                    ┌─────────────────────────────────────────────┐
                    │             INPUT MEDIA                      │
                    │  video.mp4 (or .avi / .mov / .mkv / .webm)  │
                    └──────────────┬──────────────────────────────┘
                                   │
               ┌───────────────────┼────────────────────────┐
               ▼                                            ▼
    ┌──────────────────────┐              ┌──────────────────────────┐
    │   FRAME EXTRACTION   │              │   AUDIO EXTRACTION       │
    │  RetinaFace detection│              │  FFmpeg → 16 kHz mono    │
    │  16 frames @ 224×224 │              │  librosa mel-spectrogram │
    │  face-cropped        │              │  16 segs × 128 mel × 32t │
    └──────────┬───────────┘              └────────────┬─────────────┘
               │                                       │
               ▼                                       ▼
    ┌──────────────────────┐              ┌──────────────────────────┐
    │   VISUAL ENCODER     │              │   AUDIO ENCODER          │
    │  ResNet18 (ImageNet) │              │  3-layer CNN             │
    │  per-frame features  │              │  Conv2d → ReLU → MaxPool │
    │  → 512-dim per frame │              │  → AdaptiveAvgPool       │
    └──────────┬───────────┘              │  → FC → 256-dim vector   │
               │                          └────────────┬─────────────┘
               ▼                                       │
    ┌──────────────────────┐                           │
    │  TEMPORAL TRANSFORMER│                           │
    │  (+ Frame Dropout    │                           │
    │   p=0.2 during train)│                           │
    │  2-layer Transformer │                           │
    │  d_model=512, 8 heads│                           │
    │  → 512-dim vector    │                           │
    └──────────┬───────────┘                           │
               │                                       │
               └──────────────┬────────────────────────┘
                              │
               ┌──────────────▼──────────────┐
               │   CROSS-MODAL TRANSFORMER   │
               │  Visual (Query) attends to  │
               │  Audio (Key/Value)          │
               │  8-head MultiheadAttention  │
               │  → 512-dim fused vector     │
               └──────────────┬──────────────┘
                              │
               ┌──────────────▼──────────────┐
               │      AV SYNC FEATURE        │
               │  Cosine similarity between  │
               │  projected V and A vectors  │
               │  → 1-dim scalar             │
               └──────────────┬──────────────┘
                              │
               ┌──────────────▼──────────────┐
               │    CLASSIFIER HEAD (MLP)    │
               │  Linear(513 → 512) + ReLU   │
               │  + Dropout(0.3)             │
               │  + Linear(512 → 4)          │
               └──────────────┬──────────────┘
                              │
               ┌──────────────▼──────────────┐
               │   SOFTMAX → 4-class output  │
               │  [RR, FR, RF, FF]           │
               └─────────────────────────────┘
```

### Component Details

**1. Visual Encoder**  
A ResNet18 pretrained on ImageNet has its classification head removed, leaving a feature extractor that maps each 224×224 RGB frame to a 512-dimensional embedding. All 16 frames are processed in parallel via a batch reshape (`B×T` → `B*T`), then reshaped back to `[B, T, 512]`.

**2. Temporal Transformer with Frame Dropout**  
The sequence of 16 frame embeddings is passed through a 2-layer Transformer Encoder (`d_model=512`, 8 attention heads). During training only, a `TemporalFrameDropout` module randomly zeros out entire frame embeddings with probability `p=0.2`, forcing the model to not over-rely on any single frame. The sequence is mean-pooled to produce a single 512-dim visual representation.

**3. Audio Encoder**  
A 3-layer 2D CNN processes the mel-spectrogram segments (`16 segs × 1 × 128 × 32`): Conv2d(1→32) → ReLU → MaxPool2d(2) → Conv2d(32→64) → ReLU → MaxPool2d(2) → Conv2d(64→128) → ReLU → AdaptiveAvgPool2d(1,1). The outputs are mean-pooled across segments and projected via a linear layer to a 256-dim audio embedding.

**4. Cross-Modal Transformer**  
Visual and audio embeddings are projected to a common 512-dim space. The visual features serve as the **Query** and audio features serve as the **Key** and **Value** in an 8-head cross-attention operation. This allows the model to ask: *"which visual characteristics are most associated with this audio?"*, capturing modality-specific manipulation cues. The output is a 512-dim fused embedding.

**5. AV Sync Feature**  
Both the visual and audio embeddings are L2-normalised and their dot product (cosine similarity) is computed, yielding a single scalar. A high cosine similarity indicates that audio and visual signals are coherent; a low value signals desynchronisation — a key indicator of deepfakes where audio and lip movements diverge. This scalar is concatenated with the fused embedding before classification, explicitly encoding an audio-visual synchrony signal.

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr = 1e-4 |
| Loss Function | Focal Loss (γ = 2.0) + Label Smoothing (ε = 0.1) |
| Epochs | 10 (early stopping on validation accuracy) |
| Audio Dropout | p = 0.3 |
| Frame Dropout | p = 0.2 |
| Mixed Precision | AMP (fp16) |
| Gradient Clipping | Max norm 1.0 |
| Batch Sampling | Class-balanced across all three datasets |

**Focal Loss** (γ = 2.0) down-weights easy negative examples, concentrating learning on hard-to-distinguish manipulations. **Label Smoothing** prevents overconfident predictions. **Mixed Precision (fp16)** halves memory usage and doubles throughput on compatible GPUs. The joint training across all three datasets with balanced sampling ensures the model generalises across manipulation types and compression qualities.

---

## 📊 Results

### Cross-Dataset Performance

| Dataset | Accuracy | Macro-AUC | EER |
|---------|----------|-----------|-----|
| FakeAVCeleb | **90.4%** | **0.980** | 0.9% |
| LAV-DF | 86.3% | 0.949 | **0.0%** |
| FaceForensics++ C23 | 83.1% | 0.852 | 17.9% |
| **Average** | **86.6%** | — | — |

### Per-Class F1 Scores (FakeAVCeleb)

| Class | F1 Score |
|-------|----------|
| Real Video / Real Audio | 92.9% |
| Fake Video / Real Audio | 89.3% |
| Real Video / Fake Audio | 89.6% |
| Fake Video / Fake Audio | 91.2% |

The near-perfect EER of 0.0% on LAV-DF demonstrates that the model does not trade false positives for true positives — it almost never incorrectly labels a genuine sample as fake at the optimal operating threshold.

---

## 🔍 Explainability (XAI)

Three complementary XAI techniques are computed after every inference to make the decision transparent and auditable:

### 1. Grad-CAM (Gradient-weighted Class Activation Mapping)
Gradients of the predicted class score with respect to the final convolutional feature maps of ResNet18 are computed and spatially averaged. The result is a heatmap overlaid on each frame, highlighting the facial regions (e.g. lips, eyes, jawline boundaries) that most influenced the classification. A face-swap forgery typically lights up the face boundary; a genuine sample shows diffuse, low-magnitude activation.

### 2. Temporal Attention Weights
The output attention weights from the Temporal Transformer encoder are averaged across heads and layers to produce a per-frame importance score. Frames where manipulation is most conspicuous (e.g. a mouth movement inconsistent with the audio, or a blending artefact on a specific frame) score highest. This allows an investigator to fast-forward directly to the most suspicious moments.

### 3. Audio Saliency Maps
The root-mean-square energy of each mel-spectrogram segment is computed and normalised. Segments with anomalously high energy deviation relative to neighbours flag regions where synthesised voice or spliced audio may be present.

### LLM Forensic Report (Gemini 2.5 Flash Lite)
All three evidence streams — Grad-CAM peaks, temporal attention hot frames, audio anomaly segments, AV sync score, and class probabilities — are structured into a natural-language prompt and passed to **Gemini 2.5 Flash Lite**. The model generates a 3-paragraph professional forensic report (Visual Analysis → Audio Analysis → Sync & Overall Assessment) concluding with a bolded **Final Verdict** sentence. This converts an opaque neural network decision into verifiable, auditable evidence suitable for courtroom, journalistic, or content-moderation use.

---

## 🌐 Web UI

`Deepfake_WebUI_2_1_.ipynb` launches a full-stack web application directly from Google Colab using **Flask** as the backend and **ngrok** as the public tunnel. No server infrastructure or deployment is required — run the notebook in Colab and share the generated public URL with anyone.

### Architecture

```
User Browser
     │
     │  HTTPS (public ngrok URL)
     ▼
┌────────────────────────────────────────────────────────┐
│                    NGROK TUNNEL                         │
│  (tunnels public HTTPS → localhost:5000)               │
└────────────────────────┬───────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────┐
│               FLASK APPLICATION (port 5000)             │
│                                                        │
│  GET  /           → Serves the full HTML single-page   │
│                     app (dark-themed UI with metrics   │
│                     dashboard and upload widget)       │
│                                                        │
│  POST /analyse    → Receives video + Gemini API key    │
│                     Validates: format, size (<200 MB)  │
│                     Spawns daemon inference thread     │
│                     Returns {job_id} immediately       │
│                                                        │
│  GET  /status/<id>→ Polls job status: running/done/err │
│                     Returns full result JSON on done   │
└────────────────────────────────────────────────────────┘
                         │
          Daemon thread (non-blocking)
                         │
┌────────────────────────▼───────────────────────────────┐
│               INFERENCE PIPELINE                        │
│  1. extract_frames()   — RetinaFace + 16-frame crop    │
│  2. extract_audio()    — FFmpeg → 16 kHz WAV           │
│  3. process_audio()    — librosa mel-spectrogram       │
│  4. model forward()    — MultimodalModel inference     │
│  5. compute_gradcam()  — Grad-CAM over ResNet18 conv   │
│  6. compute_temporal_attention() — Transformer attn   │
│  7. compute_audio_saliency() — RMS energy per segment  │
│  8. generate_xai_image() — matplotlib composite PNG   │
│  9. explain_with_gemini() — LLM forensic prose report  │
│  10. Return JSON result dict                           │
└────────────────────────────────────────────────────────┘
```

### UI Features

The single-page HTML application served by Flask includes:

- **📊 Project Metrics Dashboard** — Displays published accuracy, Macro-AUC, EER and per-class F1 scores directly from the paper, rendered as interactive cards and progress bars on page load.
- **📤 Video Upload Widget** — Drag-and-drop or click-to-upload area for video files (MP4, AVI, MOV, MKV, WebM), with a 200 MB size limit enforced both client-side and server-side.
- **🔑 Gemini API Key Input** — The user provides their own Gemini API key in the UI; the server never stores it beyond the request lifetime.
- **⏳ Async Job Polling** — After upload, the browser polls `/status/<job_id>` every 2 seconds. A live progress indicator shows the current inference stage without blocking the UI thread.
- **🧬 Results Panel (tabbed)**:
  - **Detection Tab** — Verdict banner (✅ or 🚨), confidence %, filename, processing time, and four metric chips (Video Fake Prob, Audio Fake Prob, AV Sync Mismatch, Processing Time) colour-coded green/red by threshold. Four probability bars (one per class) show the full softmax distribution.
  - **XAI Tab** — A dark-themed composite matplotlib figure showing: 8 original frames (top row), 8 Grad-CAM overlays (middle row), a temporal attention bar chart per frame (bottom-left), and an audio segment saliency area chart (bottom-right). Clicking the XAI image zooms it to full screen.
  - **Report Tab** — The full Gemini-generated forensic report in plain prose.
- **⚠️ Audio-Only Warning** — If the uploaded video has no audio track, the UI displays a warning badge and the model runs in video-only mode.
- **Hard Timeout** — Each inference job is given a 60-second timeout enforced in the worker thread; any stage that exceeds the limit returns a clean error to the UI.

### How to Launch

1. Open `Deepfake_WebUI_2_1_.ipynb` in Google Colab.
2. Set the runtime to **GPU (T4 recommended)**.
3. Mount your Google Drive (Cell 2) — the trained checkpoint must be at `MyDrive/Deepfake/joint_model_checkpoint.pth`.
4. Set your ngrok auth token in Cell 3 (get one free at [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken)).
5. Run all cells in order (Cells 1–6).
6. Copy the printed **PUBLIC URL** (e.g. `https://grumble-backed-sublease.ngrok-free.dev`) and open it in any browser.
7. Enter your [Gemini API key](https://aistudio.google.com/app/apikey), upload a video, and click **Analyse**.

---

## 🚀 Getting Started (Full Pipeline)

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install librosa retina-face opencv-python ffmpeg-python
pip install numpy matplotlib scikit-learn Pillow
pip install google-generativeai
pip install flask flask-cors pyngrok
```

### Step 1 — Preprocess Datasets

Run each preprocessing notebook for the datasets you have downloaded:

```
FakeAVCeleb_Preprocess.ipynb
LAV_DF_Preprocess.ipynb
FF_C23_Preprocess.ipynb
```

Each notebook performs the following steps:
- **Face Detection**: RetinaFace detects faces in the first 30 frames; the largest detected bounding box is fixed and applied to all subsequent frames for stability.
- **Frame Sampling**: 16 frames are uniformly sampled across the video duration. LAV-DF uses timestamp-aware sampling, preferring frames within annotated manipulation intervals to ensure meaningful training signal from its sparse (~9%) fake frames.
- **Audio Extraction**: FFmpeg extracts mono audio at 16 kHz.
- **Mel-Spectrogram**: librosa generates 16 non-overlapping segments of 128 mel bins × 32 time steps.
- **Output**: Each video is saved as a compressed `.npz` archive containing the frame tensor and audio tensor.
- **Splits**: FakeAVCeleb splits 80/20 by speaker identity (preventing the same speaker appearing in both train and test). FF++ splits 80/20 by manipulation type.

### Step 2 — Train the Model

```
Joint_Training.ipynb
```

Trains the multimodal fusion model jointly across all three datasets using class-balanced sampling. The notebook handles dataset loading from the `.npz` archives, sets up the Focal Loss + Label Smoothing composite loss, runs mixed-precision training with gradient clipping, and saves the best checkpoint (by average validation accuracy) to Google Drive.

### Step 3 — Run Inference (Notebook Mode)

```
Inference.ipynb
```

Runs the trained model on test-split samples, generates Grad-CAM overlays, temporal attention bar charts, audio saliency visualisations, and calls the Gemini API to produce a full LLM forensic report for each sample. Results are saved as PNG visualisation files.

### Step 4 — Launch the Web UI

```
Deepfake_WebUI_2_1_.ipynb
```

See the [Web UI section](#-web-ui-new) above for full instructions.

---

## 📋 Preprocessing Details Summary

| Stage | Detail |
|-------|--------|
| Face Detector | RetinaFace — selects largest face; bounding box fixed from first 30 frames |
| Frame Resolution | 224 × 224 px (centre-cropped from face bounding box) |
| Frames per Video | 16 (uniform sampling); LAV-DF uses manipulation-interval-aware sampling |
| Audio Sample Rate | 16 kHz mono (FFmpeg extraction) |
| Mel Segments | 16 segments of 128 mel bins × 32 time steps (via librosa) |
| Train/Test Splits | FakeAVCeleb: 80/20 by speaker identity; FF++: 80/20 by manipulation type |
| Storage Format | Per-video `.npz` archives (frames + audio tensors) |

---

## 📜 Citation

If you use this work, please cite the conference paper:

```
Balasudhan C M, Thilak Raaj N V, Samuel Raj Irwin V, Sujatha K.
"Multimodal Deepfake Detection with Cross-Modal Attention and Explainable AI."
Proceedings of ICRETEM 2026, Suguna College of Engineering, Coimbatore, India.
April 10–11, 2026.
```

---

## 📜 License

This project is released for academic and research purposes. Please contact the authors for any other use.
