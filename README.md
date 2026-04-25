# 💊 Tablet Defect Inspector AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-EE4C2C?style=for-the-badge&logo=pytorch)
![Gradio](https://img.shields.io/badge/Gradio-App-FF7C00?style=for-the-badge&logo=gradio)
![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-green?style=for-the-badge)

### 🚀 AI-powered pharmaceutical quality control — detect defective tablets instantly!

**[🌐 Try the Live Demo »](https://huggingface.co/spaces/BUDDDY2894830/tablet-defect-inspector)**

</div>

---

## ✨ What is this?

An **AI-powered tablet/pill quality inspection system** built with deep learning. Upload a photo of a tablet and the AI instantly tells you whether it's **good or defective** — with confidence scores!

> 🏭 Built for pharmaceutical quality control, this tool can help factories detect cracked, contaminated, or damaged tablets before they reach patients.

---

## 🎯 Features

| Feature | Details |
|---|---|
| 🧠 **AI Model** | Fine-tuned ResNet18 (Deep Learning) |
| ⚡ **Instant Results** | Real-time prediction with confidence scores |
| 📊 **Confidence Scores** | Shows probability for each class |
| 📋 **Inspection History** | Tracks all inspections in a session |
| 📷 **Live Camera Mode** | Real-time webcam/phone inspection |
| 🔴 **Defect Alarm** | Audio alert when defective tablet detected |
| 🌐 **Web App** | Beautiful Gradio interface |

---

## 🔍 Detection Classes

```
✅  GOOD      — Tablet is intact and safe
❌  DEFECTIVE — Tablet is cracked or contaminated
```

---

## 🛠️ Tech Stack

```
🐍 Python          — Core language
🔥 PyTorch         — Deep learning framework
🖼️  ResNet18        — Pre-trained CNN (transfer learning)
🎨 Gradio          — Web app interface
📦 Torchvision     — Image transforms & model zoo
📷 OpenCV          — Real-time camera feed
🔊 Pygame          — Defect alarm sound
```

---

## 📁 Project Structure

```
tablet-defect-inspector/
│
├── 📄 app.py              # Gradio web application
├── 📄 train.py            # Model training script
├── 📄 realtime.py         # Live webcam inspection
├── 🧠 tablet_model.pth    # Trained ResNet18 model (~44MB)
├── 📋 requirements.txt    # Python dependencies
├── 📝 README.dataset.txt  # Dataset information
└── 📝 README.roboflow.txt # Roboflow export info
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/mabdullahab614-alt/tablet-defect-inspector.git
cd tablet-defect-inspector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the web app
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860` 🎉

---

## 📷 Real-Time Inspection (Webcam/Phone)

```bash
python realtime.py
```

> 📱 You can stream from your **phone camera** using IP Webcam app!
> Edit `PHONE_IP` in `realtime.py` to your phone's IP address.

---

## 🏋️ Train Your Own Model

```bash
# Organize your dataset like this:
# dataset/train/*.jpg + *.png (masks)
# dataset/valid/*.jpg + *.png (masks)

python train.py
```

Training runs for **5 epochs** and saves the best model automatically.

---

## 📊 Dataset

- **Source:** [Roboflow Universe — Pill Dataset](https://universe.roboflow.com/abdullah-javid/pill-kjhgx-ac03p)
- **Size:** 959 images
- **Format:** PNG Masks for Semantic Segmentation
- **License:** CC BY 4.0

---

## 🌐 Live Demo

Try it right now — no installation needed!

<div align="center">

[![🚀 Launch Live Demo](https://img.shields.io/badge/🚀%20LAUNCH%20LIVE%20DEMO-Click%20Here!-FF4B4B?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/BUDDDY2894830/tablet-defect-inspector)

👉 **[https://huggingface.co/spaces/BUDDDY2894830/tablet-defect-inspector](https://huggingface.co/spaces/BUDDDY2894830/tablet-defect-inspector)**

</div>

---

## 👨‍💻 Author

**Abdullah Javid**

[![GitHub](https://img.shields.io/badge/GitHub-mabdullahab614--alt-black?style=flat&logo=github)](https://github.com/mabdullahab614-alt)
[![Hugging Face](https://img.shields.io/badge/🤗-BUDDDY2894830-yellow?style=flat)](https://huggingface.co/BUDDDY2894830)

---

## 📄 License

This project is licensed under **CC BY 4.0** — feel free to use, share, and adapt with attribution.

---

<div align="center">

⭐ **If you found this useful, please star the repo!** ⭐

*Built with ❤️ using PyTorch & Gradio*

</div>
