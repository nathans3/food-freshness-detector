---
title: Food Freshness Detector
emoji: 🍎
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.20.0
app_file: app.py
pinned: false
---

# Food Freshness Detector

A FastAI + Gradio image classifier that predicts the freshness of fruits and vegetables.

Upload a photo and it will classify the food as one of:
- **Fresh** — looks good, ready to eat
- **Slightly Spoiled** — shows mild spoilage signs; inspect before eating
- **Rotten** — likely spoiled and unsafe to consume

## Model

- Architecture: ResNet18 (transfer learning via FastAI)
- Input: RGB image, resized to 224×224
- Output: 3-class softmax probabilities

## Files

- `app.py` — Gradio inference app
- `requirements.txt` — Python dependencies
- `model/config.json` — architecture config (`arch`, `n_classes`, `img_size`)
- `model/vocab.json` — class labels
- `model/model_weights.pth` — trained ResNet18 weights (~45 MB)
- `assets/demo.png` — example image
