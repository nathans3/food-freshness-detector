import json
import os
from pathlib import Path

import gradio as gr
import torch
from fastai.vision.all import create_cnn_model, resnet18
from PIL import Image

MODEL_DIR = Path("model")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "nathansekar/food-freshness-detector")

CLASS_DESCRIPTIONS = {
    "fresh": "Looks fresh and ready to eat.",
    "slightly_spoiled": "Shows mild spoilage signs; inspect before eating.",
    "rotten": "Likely spoiled and unsafe to consume.",
}


def download_model_files() -> None:
    """Download model weights from HF model repo if not already present locally."""
    weights_path = MODEL_DIR / "model_weights.pth"
    if weights_path.exists():
        return

    try:
        from huggingface_hub import hf_hub_download

        print(f"Downloading model weights from {HF_MODEL_REPO}…")
        MODEL_DIR.mkdir(exist_ok=True)

        for filename in ("model_weights.pth", "config.json", "vocab.json"):
            dest = MODEL_DIR / filename
            if not dest.exists():
                downloaded = hf_hub_download(repo_id=HF_MODEL_REPO, filename=filename)
                import shutil
                shutil.copy(downloaded, dest)
                print(f"  {filename} ready")
    except Exception as exc:
        print(f"WARNING: Could not download model from HF: {exc}")


download_model_files()

learner = None
load_error = None
vocab = []

if (MODEL_DIR / "model_weights.pth").exists():
    try:
        with open(MODEL_DIR / "config.json") as f:
            config = json.load(f)
        with open(MODEL_DIR / "vocab.json") as f:
            vocab = json.load(f)

        model = create_cnn_model(resnet18, config["n_classes"])
        model.load_state_dict(
            torch.load(MODEL_DIR / "model_weights.pth", map_location="cpu")
        )
        model.eval()
        learner = model
    except Exception as exc:
        load_error = str(exc)
else:
    load_error = (
        "Model weights not found. "
        "Set HF_MODEL_REPO to your Hugging Face model repo ID."
    )


def predict_freshness(image: Image.Image):
    if learner is None:
        return (
            "Model unavailable",
            {"fresh": 0.0, "slightly_spoiled": 0.0, "rotten": 0.0},
            load_error or "Unknown model loading error.",
        )

    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = learner(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    pred_class = vocab[pred_idx]

    confidence = {vocab[i]: float(probs[i]) for i in range(len(probs))}
    confidence = dict(sorted(confidence.items(), key=lambda x: x[1], reverse=True))

    description = CLASS_DESCRIPTIONS.get(pred_class, "No class description available.")

    return pred_class, confidence, description


title = "Food Freshness Detector"
description = (
    "Upload a fruit or vegetable photo to classify it as Fresh, Slightly Spoiled, or Rotten."
)

with gr.Blocks(theme=gr.themes.Soft(), title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Food Image")

    with gr.Row():
        label_output = gr.Label(label="Confidence Scores")

    with gr.Row():
        pred_text = gr.Textbox(label="Predicted Class")
        details_text = gr.Textbox(label="Interpretation")

    predict_btn = gr.Button("Predict")

    predict_btn.click(
        fn=predict_freshness,
        inputs=image_input,
        outputs=[pred_text, label_output, details_text],
    )

    gr.Examples(
        examples=["assets/demo.png"],
        inputs=image_input,
        label="Try Example",
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
