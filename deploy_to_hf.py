"""
Deploy Food Freshness Detector model to a Hugging Face Model repository (free).

This uploads the trained weights and metadata to a model repo at:
  https://huggingface.co/USERNAME/food-freshness-detector

Model repos are always free on Hugging Face (no Pro subscription required).

Usage:
    python deploy_to_hf.py --username YOUR_HF_USERNAME --token YOUR_HF_TOKEN

Required local files:
  - model/model_weights.pth  (~45 MB — uploaded via Git LFS automatically)
  - model/config.json
  - model/vocab.json
  - assets/demo.png
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
MODEL_REPO_NAME = "food-freshness-detector"

MODEL_CARD = """\
---
language: en
license: mit
tags:
  - image-classification
  - food
  - freshness
  - fastai
  - pytorch
  - resnet
library_name: fastai
---

# Food Freshness Detector

A ResNet18 image classifier trained with FastAI that predicts the freshness of
fruits and vegetables.

**Classes:** Fresh · Slightly Spoiled · Rotten

## Usage

```python
import json
import torch
from fastai.vision.all import create_cnn_model, resnet18
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# Download weights
weights_path = hf_hub_download("USERNAME/food-freshness-detector", "model_weights.pth")
config_path  = hf_hub_download("USERNAME/food-freshness-detector", "config.json")
vocab_path   = hf_hub_download("USERNAME/food-freshness-detector", "vocab.json")

with open(config_path) as f:
    config = json.load(f)
with open(vocab_path) as f:
    vocab = json.load(f)

model = create_cnn_model(resnet18, config["n_classes"])
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("your_food_image.jpg").convert("RGB")
tensor = preprocess(image).unsqueeze(0)

with torch.no_grad():
    probs = torch.softmax(model(tensor), dim=1)[0]

pred_idx = probs.argmax().item()
print(f"Prediction: {vocab[pred_idx]} ({probs[pred_idx]:.1%})")
```

## Model Details

| | |
|---|---|
| Architecture | ResNet18 (transfer learning) |
| Framework | FastAI 2.x / PyTorch |
| Input | RGB image → 224×224 |
| Output | 3-class softmax |
| Training data | Synthetic fruit/vegetable images |

## Files

| File | Description |
|---|---|
| `model_weights.pth` | Trained ResNet18 state dict |
| `config.json` | Architecture config (`arch`, `n_classes`, `img_size`) |
| `vocab.json` | Class label list |
"""

REQUIRED_FILES = [
    REPO_ROOT / "model" / "model_weights.pth",
    REPO_ROOT / "model" / "config.json",
    REPO_ROOT / "model" / "vocab.json",
    REPO_ROOT / "assets" / "demo.png",
]


def check_required_files() -> bool:
    missing = [f for f in REQUIRED_FILES if not f.exists()]
    if not missing:
        return True

    print("ERROR: The following required files are missing:")
    for f in missing:
        print(f"  {f.relative_to(REPO_ROOT)}")
    print("\nRun train.ipynb first to generate model_weights.pth.")

    return False


def deploy(username: str, token: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    repo_id = f"{username}/{MODEL_REPO_NAME}"
    api = HfApi(token=token)

    print(f"Creating / verifying model repo: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=False,
    )
    print("  Model repo ready.")

    # Write model card with the correct username substituted in
    card_content = MODEL_CARD.replace("USERNAME", username)
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print("  Uploaded README.md (model card)")

    uploads = [
        (REPO_ROOT / "model" / "model_weights.pth", "model_weights.pth"),
        (REPO_ROOT / "model" / "config.json",        "config.json"),
        (REPO_ROOT / "model" / "vocab.json",          "vocab.json"),
        (REPO_ROOT / "assets" / "demo.png",           "assets/demo.png"),
    ]

    for local_path, repo_path in uploads:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  Uploading {repo_path} ({size_mb:.1f} MB)…", end="", flush=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
        )
        print(" done")

    model_url = f"https://huggingface.co/{repo_id}"
    print(f"\nModel repo live: {model_url}")
    print("\nNext step: deploy the Gradio app to Render.")
    print("  1. Push this repo to GitHub.")
    print("  2. Go to https://render.com → New Web Service → connect your repo.")
    print("  3. Render will pick up render.yaml automatically.")
    print("  4. Set the env var HF_MODEL_REPO=" + repo_id + " in Render's dashboard.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload Food Freshness Detector model to Hugging Face (free model repo)."
    )
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--token",    required=True, help="Your Hugging Face write token")
    args = parser.parse_args()

    if not check_required_files():
        sys.exit(1)

    deploy(args.username, args.token)


if __name__ == "__main__":
    main()
