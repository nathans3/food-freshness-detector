from pathlib import Path

import gradio as gr
from fastai.learner import load_learner
from fastai.vision.core import PILImage
from PIL import Image

MODEL_PATH = Path("model/export.pkl")
CLASS_DESCRIPTIONS = {
    "fresh": "Looks fresh and ready to eat.",
    "slightly_spoiled": "Shows mild spoilage signs; inspect before eating.",
    "rotten": "Likely spoiled and unsafe to consume.",
}

learner = None
load_error = None
if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
    try:
        learner = load_learner(MODEL_PATH)
    except Exception as exc:
        load_error = str(exc)
else:
    load_error = (
        "Model file not found or empty at model/export.pkl. "
        "Run train.ipynb and export the learner first."
    )


def predict_freshness(image: Image.Image):
    if learner is None:
        return (
            "Model unavailable",
            {"fresh": 0.0, "slightly_spoiled": 0.0, "rotten": 0.0},
            load_error or "Unknown model loading error.",
        )

    # Save temp file and load via path (avoids PILImage conversion issues)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image.save(tmp.name)
        fastai_image = PILImage.create(tmp.name)
    
    pred_class, pred_idx, probs = learner.predict(fastai_image)
    pred_class = str(pred_class)

    confidence = {learner.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}
    confidence = dict(sorted(confidence.items(), key=lambda item: item[1], reverse=True))

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
    demo.launch()
