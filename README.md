# Food Freshness Detector

A complete **FastAI (Chapters 1-4 style)** computer vision project that classifies fruit/vegetable images into:

- **Fresh**
- **Slightly Spoiled**
- **Rotten**

This project is designed as a portfolio-ready repository for students learning transfer learning with `fastai`.

## Features

- End-to-end workflow with `DataBlock` and `DataLoaders`
- Data augmentation via `aug_transforms()`
- Transfer learning using `vision_learner()` + `resnet34`
- Learning rate discovery with `lr_find()`
- Training with `fine_tune()`
- Evaluation using confusion matrix and top losses
- Exported model (`export.pkl`) for inference
- Gradio web app for local and Hugging Face deployment

## Model Architecture

- **Backbone**: `resnet34`
- **Framework**: `fastai` vision API
- **Training style**: transfer learning with progressive unfreezing through `fine_tune()`
- **Metrics**: `accuracy`, `error_rate`

## Dataset

The notebook assumes this structure:

```text
data/food_freshness/
├── fresh/
├── slightly_spoiled/
└── rotten/
```

See `data/README.md` for dataset source suggestions, download guidance, and quality checks.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Open and run the notebook:

```bash
jupyter notebook train.ipynb
```

At the end of training, the notebook exports the model to:

```text
model/export.pkl
```

## Running Locally

After `export.pkl` exists:

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal.

## Deploying to Hugging Face Spaces

1. Create a new **Gradio** Space.
2. Push this repository to GitHub.
3. Connect the repository to your Space (or upload files manually).
4. Ensure these files are present in root: `app.py`, `requirements.txt`, `model/export.pkl`.
5. Build and test the app.

Use `huggingface/README.md` as the Space card template.

## Results

You will generate these in `train.ipynb`:

- Class distribution checks
- Sample augmented batches
- Validation metrics
- Confusion matrix
- Top-loss examples

Add screenshots to `assets/screenshots/` after training for a polished portfolio.

## Example Predictions

- Clear bright apple with smooth skin → **Fresh**
- Banana with dark spots but mostly intact peel → **Slightly Spoiled**
- Moldy strawberry with collapse/discoloration → **Rotten**

## Future Improvements

- Expand to more food categories
- Add confidence threshold warnings
- Mobile-first camera UI
- Grad-CAM visual explanations
- Freshness + estimated shelf life
- Nutrition-aware recommendations

## Acknowledgements

- FastAI course and book: *Deep Learning for Coders with fastai & PyTorch*
- Jeremy Howard and the FastAI community
- Public dataset contributors

## License

Released under the MIT License. See `LICENSE`.
