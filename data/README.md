# Dataset Instructions

This folder stores **instructions only**. Do not commit large image files to GitHub.

## Recommended Dataset Strategy (Chapter 1-4 Friendly)

Use one combined dataset with these 3 labels:

- `fresh`
- `slightly_spoiled`
- `rotten`

### Candidate sources

1. **Kaggle** datasets tagged with fruit/vegetable freshness or rotten/fresh classification.
2. Your own manually curated images from search (FastAI Chapter 2 style), then cleaned with `ImageClassifierCleaner`.

A strong beginner approach:

- Start with a public rotten-vs-fresh fruit dataset.
- Add a third class (`slightly_spoiled`) by collecting edge-case images manually.
- Clean labels in notebook before final training.

## Expected Folder Layout

After downloading/curating, organize data as:

```text
data/
└── food_freshness/
    ├── fresh/
    │   ├── img001.jpg
    │   └── ...
    ├── slightly_spoiled/
    │   ├── img201.jpg
    │   └── ...
    └── rotten/
        ├── img401.jpg
        └── ...
```

## Quick Quality Checklist

- Keep image sizes reasonably varied; FastAI handles resizing.
- Avoid watermark-heavy and meme-like images.
- Include multiple produce types (apple, banana, tomato, orange, strawberry).
- Ensure similar counts per class (imbalance hurts learning).
- Remove corrupted files with `verify_images`.

## Notes

The training notebook (`train.ipynb`) assumes the dataset path:

```python
path = Path('data/food_freshness')
```

Change this path if your local structure differs.
