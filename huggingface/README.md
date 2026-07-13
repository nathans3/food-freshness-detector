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

# Food Freshness Detector (Hugging Face Space)

This Space hosts a FastAI + Gradio image classifier for food freshness:

- Fresh
- Slightly Spoiled
- Rotten

## Required Files in Space Root

- `app.py`
- `requirements.txt`
- `model/export.pkl`
- `assets/demo.png` (optional example image)

## Deploy Steps

1. Create a new **Gradio Space** on Hugging Face.
2. Upload repository files (or connect your GitHub repo).
3. Ensure `model/export.pkl` is present and non-empty.
4. Wait for build to finish.
5. Test with sample fruit/vegetable images.

## Common Issues

- If build fails, verify pinned package versions in `requirements.txt`.
- If app loads but predicts nothing, check `model/export.pkl` compatibility with current fastai version.
- If classes look incorrect, verify your training folder names and export file.
