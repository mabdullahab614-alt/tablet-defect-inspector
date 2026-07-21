<div align="center">

<!-- Animated Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:f59e0b,40:d97706,100:92400e&height=220&section=header&text=💊%20Tablet%20Defect%20Inspector&fontSize=50&fontColor=ffffff&animation=twinkling&fontAlignY=42&desc=ResNet18%20·%20Good%20vs%20Defective%20·%20Real-Time%20Webcam%20·%20Defect%20Alarm%20·%20959%20Images&descAlignY=65&descSize=15&descColor=fef3c7" width="100%"/>

<br/>

<!-- CTA Button -->
<a href="https://huggingface.co/spaces/BUDDDY2894830/tablet-defect-inspector">
  <img src="https://img.shields.io/badge/▶%20%20T%20R%20Y%20%20L%20I%20V%20E%20%20N%20O%20W-f59e0b?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=451a03" height="52" alt="Try Live"/>
</a>

<br/><br/>

<!-- Badge Row 1 -->
<a href="https://huggingface.co/spaces/BUDDDY2894830/tablet-defect-inspector"><img src="https://img.shields.io/badge/Model-ResNet18-F59E0B?style=for-the-badge&labelColor=0f172a"/></a>
&nbsp;
<a href="https://huggingface.co/spaces/BUDDDY2894830/tablet-defect-inspector"><img src="https://img.shields.io/badge/Task-Binary%20Classification-22C55E?style=for-the-badge&labelColor=0f172a"/></a>
&nbsp;
<a href="https://universe.roboflow.com/abdullah-javid/pill-kjhgx-ac03p"><img src="https://img.shields.io/badge/Dataset-959%20Images-3B82F6?style=for-the-badge&labelColor=0f172a"/></a>
&nbsp;
<a href="#-license"><img src="https://img.shields.io/badge/License-All%20Rights%20Reserved-DC2626?style=for-the-badge&labelColor=0f172a"/></a>

<br/><br/>

<!-- Badge Row 2 -->
<a href="https://huggingface.co/spaces/BUDDDY2894830/tablet-defect-inspector"><img src="https://img.shields.io/badge/Deployed%20on-Hugging%20Face%20Spaces-FF6B00?style=for-the-badge&logo=huggingface&labelColor=0f172a"/></a>
&nbsp;
<a href="https://huggingface.co/spaces/BUDDDY2894830/tablet-defect-inspector"><img src="https://img.shields.io/badge/Interface-Gradio-F97316?style=for-the-badge&labelColor=0f172a"/></a>
&nbsp;
<a href="https://github.com/mabdullahab614-alt/tablet-defect-inspector"><img src="https://img.shields.io/badge/Framework-PyTorch-EF4444?style=for-the-badge&logo=pytorch&labelColor=0f172a"/></a>
&nbsp;
<a href="https://github.com/mabdullahab614-alt/tablet-defect-inspector"><img src="https://img.shields.io/badge/Webcam-Real--Time%20Mode-06B6D4?style=for-the-badge&labelColor=0f172a"/></a>

</div>

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:f59e0b,50:d97706,100:92400e&height=3" width="100%"/>

<br/>

<div align="center">

## 🧠 INSPECTION PIPELINE ARCHITECTURE

```
╔══════════════════════════════════════════════════════════════════╗
║        TABLET DEFECT INSPECTOR — RESNET18 QC PIPELINE           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   INPUT: Tablet Photo  OR  Live Webcam Frame                    ║
║      │                                                           ║
║      ▼                                                           ║
║   ┌─────────────────────────────────────────┐                    ║
║   │         IMAGE PRE-PROCESSING            │                    ║
║   │   • Resize to 224×224 px                │                    ║
║   │   • Normalize (ImageNet mean/std)        │                    ║
║   │   • Training augmentations:             │                    ║
║   │       RandomHorizontalFlip             │                    ║
║   │       RandomVerticalFlip               │                    ║
║   │       RandomRotation ±30°              │                    ║
║   │       ColorJitter (brightness/contrast) │                    ║
║   └─────────────────────────────────────────┘                    ║
║      │                                                           ║
║      ▼                                                           ║
║   ┌─────────────────────────────────────────┐                    ║
║   │        RESNET18 BACKBONE                │                    ║
║   │   Pre-trained on ImageNet (1M+ images)  │                    ║
║   │                                         │                    ║
║   │   Conv1 → BN → ReLU → MaxPool          │                    ║
║   │   Layer1 [64]  → Layer2 [128]          │                    ║
║   │   Layer3 [256] → Layer4 [512]          │                    ║
║   │   Residual skip connections             │                    ║
║   └─────────────────────────────────────────┘                    ║
║      │                                                           ║
║      ▼                                                           ║
║   ┌─────────────────────────────────────────┐                    ║
║   │      BINARY CLASSIFICATION HEAD         │                    ║
║   │   AdaptiveAvgPool → Flatten             │                    ║
║   │   Linear(512 → 2)                      │                    ║
║   │   Softmax → [GOOD%  |  DEFECTIVE%]     │                    ║
║   └─────────────────────────────────────────┘                    ║
║      │                                                           ║
║      ▼                                                           ║
║   ┌─────────────────────────────────────────┐                    ║
║   │        OUTPUT + ALARM SYSTEM            │                    ║
║   │   ✅ GOOD     → Green result card       │                    ║
║   │   ❌ DEFECTIVE → Red result + 🔴 Alarm  │                    ║
║   │   Confidence % displayed for both       │                    ║
║   │   Session history logged per inspection │                    ║
║   └─────────────────────────────────────────┘                    ║
║                                                                  ║
║   ⚡ 959 Roboflow images · 5-epoch training · Transfer Learning  ║
╚══════════════════════════════════════════════════════════════════╝
```

</div>

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:f59e0b,50:d97706,100:92400e&height=3" width="100%"/>

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔬 Detection
- ✅ **ResNet18** + Transfer Learning (ImageNet)
- ✅ **Binary classification** — Good vs Defective
- ✅ **Confidence scores** for both classes
- ✅ **Defect Alarm** — audio alert on bad tablet
- ✅ **Inspection History** — full session log
- ✅ **Instant results** — real-time inference

### 📊 Detection Classes
| Result | Meaning |
|--------|---------|
| ✅ **GOOD** | Tablet intact — safe to dispatch |
| ❌ **DEFECTIVE** | Cracked / contaminated — reject |

</td>
<td width="50%">

### 📷 Inspection Modes
- ✅ **Photo Upload** — single image inspection
- ✅ **Live Webcam** — real-time camera feed
- ✅ **Phone Camera** — IP Webcam app support
- ✅ **Gradio web UI** — zero code needed
- ✅ **Free on Hugging Face** — no install required

### 🏭 Industrial Use Case
- ✅ **Pharmaceutical QC** — pre-dispatch screening
- ✅ **Factory-floor ready** — webcam integration
- ✅ **Custom dataset** — Roboflow 959 pill images
- ✅ **Retrain easily** — `train.py` included
- ✅ **5-epoch training** — fast to fine-tune

</td>
</tr>
</table>

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:f59e0b,50:d97706,100:92400e&height=3" width="100%"/>

## 🚀 Live Demo

<div align="center">

### 🌐 [https://huggingface.co/spaces/BUDDDY2894830/tablet-defect-inspector](https://huggingface.co/spaces/BUDDDY2894830/tablet-defect-inspector)

*Upload a tablet photo — get Good or Defective verdict in under a second*

</div>

---

## 🛠 Tech Stack

<div align="center">

| | Technology | Purpose |
|--|-----------|---------|
| 🧠 | **ResNet18** (Transfer Learning) | Core binary classification backbone |
| 🔥 | **PyTorch** + **Torchvision** | Model training & inference |
| 📷 | **OpenCV** | Real-time webcam frame capture |
| 🔊 | **Pygame** | Defect alarm sound trigger |
| 🎛️ | **Gradio** | Interactive web interface |
| 🤗 | **Hugging Face Spaces** | Free cloud deployment |
| 📦 | **Roboflow** | Dataset source & annotation |
| 🐍 | **Python 3.8+** | Core language |

</div>

---

## ⚡ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/mabdullahab614-alt/tablet-defect-inspector.git
cd tablet-defect-inspector

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run the web app (photo upload)
python app.py
# → Opens at http://127.0.0.1:7860

# 3b. Run real-time webcam inspection
python realtime.py
```

### 📱 Phone Camera Mode

```python
# Edit realtime.py — set your phone's IP:
PHONE_IP = "192.168.x.x"   # from IP Webcam app
```

> Install **IP Webcam** on Android, start server, set the IP above and run `realtime.py`

### 🏋️ Train Your Own Model

```bash
# Prepare dataset:
# dataset/train/  — training images
# dataset/valid/  — validation images

python train.py   # Runs 5 epochs, saves best model
```

---

## 📊 Dataset & Performance

<div align="center">

| Metric | Value |
|--------|-------|
| Dataset Source | Roboflow Universe |
| Total Images | **959 tablet photos** |
| Training Epochs | 5 |
| Base Model | ResNet18 (ImageNet) |
| Training Method | Transfer Learning |
| Output Classes | **GOOD · DEFECTIVE** |
| Inference Time | **< 1 second** |

</div>

---

## 📁 Project Structure

```
tablet-defect-inspector/
├── app.py               # Gradio web app (photo upload)
├── realtime.py          # Live webcam / phone camera mode
├── train.py             # Model training script
├── requirements.txt     # Python dependencies
├── README.dataset.txt   # Dataset info
└── README.roboflow.txt  # Roboflow export metadata
```

---

## 🏆 Rating

<div align="center">

| Category | Score |
|----------|-------|
| Model Architecture | ⭐⭐⭐⭐⭐ |
| Industrial Relevance | ⭐⭐⭐⭐⭐ |
| Real-Time Webcam Mode | ⭐⭐⭐⭐⭐ |
| Defect Alarm System | ⭐⭐⭐⭐⭐ |
| Ease of Use | ⭐⭐⭐⭐⭐ |
| Deployment | ⭐⭐⭐⭐⭐ |
| **OVERALL** | **⭐⭐⭐⭐⭐ 10/10** |

</div>

---

## 📜 License

**All Rights Reserved © 2026 Abdullah Javid**

This repository and its contents — including source code, trained model, dataset structure, ideas, and documentation — are made publicly visible **for portfolio and demonstration purposes only**.

**No part of this repository may be copied, modified, distributed, sublicensed, or used** — in whole or in part, for personal, educational, or commercial purposes — without explicit prior written permission from the author.

Forking or cloning this repository does **not** grant any rights to use, reproduce, or redistribute its contents.

If you are interested in using any part of this project, please contact me directly for permission:

📧 **Email:** mabdullah.ab614@gmail.com
🔗 **GitHub:** [github.com/mabdullahab614-alt](https://github.com/mabdullahab614-alt)
💼 **LinkedIn:** [linkedin.com/in/abdullah-javid-b217a2384](https://www.linkedin.com/in/abdullah-javid-b217a2384/)

---

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:f59e0b,50:d97706,100:92400e&height=3" width="100%"/>

<div align="center">

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-mabdullahab614--alt-181717?style=for-the-badge&logo=github&labelColor=0f172a)](https://github.com/mabdullahab614-alt)
&nbsp;
[![Live Demo](https://img.shields.io/badge/💊%20Try%20Tablet%20Inspector-f59e0b?style=for-the-badge&labelColor=0f172a)](https://huggingface.co/spaces/BUDDDY2894830/tablet-defect-inspector)
&nbsp;
[![Portfolio](https://img.shields.io/badge/🌐%20Portfolio-Abdullah%20Javid-8B5CF6?style=for-the-badge&labelColor=0f172a)](https://portfolio-website-jet-iota-21.vercel.app/)

<br/><br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:92400e,60:d97706,100:f59e0b&height=120&section=footer&animation=twinkling" width="100%"/>

**⭐ Star this repo if it helped your QC pipeline!**

</div>
