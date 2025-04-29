---
title: PneumoX Net
emoji: 📚
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: false
license: mit
short_description: ' AI-Powered Pneumonia Detection via X-ray Image '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🩺 PneumoX-Net - Chest X-ray Pneumonia Detection Model

[![Hugging Face Spaces](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/spaces/dkg-2/PneumoX-Net)

PneumoX-Net is a deep learning-powered diagnostic tool that classifies **Pneumonia** vs **Normal** chest X-rays. It aids early screening and diagnosis, especially in low-resource settings.

---

## 🧠 Model Overview

PneumoX-Net is built on **MobileNetV2** using transfer learning, fine-tuned for binary classification (Pneumonia vs. Normal) on the NIH Chest X-ray dataset. It has demonstrated high diagnostic accuracy on unseen X-ray images.

---

## 📊 Evaluation Metrics

| Metric       | Score   |
|--------------|---------|
| **Accuracy** | 96.18%  |
| **Precision**| 97.60%  |
| **Recall**   | 99.12%  |
| **F1-Score** | 97.37%  |
| **AUC**      | 99.25%  |
| **Log Loss** | 0.1388  |

> 🔍 High recall ensures minimal false negatives — a crucial metric for healthcare models.

---

## 🛠 Technology Stack

- 🔹 **Deep Learning**: TensorFlow / Keras
- 🔹 **Pretrained Backbone**: MobileNetV2
- 🔹 **Image Processing**: OpenCV, PIL (Pillow)
- 🔹 **Deployment**: Gradio + Hugging Face Spaces
- 🔹 **Visualization**: Matplotlib, Seaborn

---

## 🔗 Resources

- 📓 **Colab Notebook** (Model Training & Evaluation):  
  👉 [Open in Colab](https://colab.research.google.com/drive/1pwfrmO31SE7bxQdCDwPcoQJqxJjpi1st?usp=sharing)

- 🧪 **Try the Model Live**:  
  👉 [Hugging Face Space](https://huggingface.co/spaces/dkg-2/PneumoX-Net)

## 💻 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/PneumoX-Net.git
cd PneumoX-Net
pip install -r requirements.txt
python app/app.py

