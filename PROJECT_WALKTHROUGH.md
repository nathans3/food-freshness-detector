# 📚 Project Walkthrough: How Everything Fits Together

## Overview

This project demonstrates the complete machine learning workflow from FastAI Chapters 1-4:
1. **Data Preparation** → Organizing images into folders by label
2. **Model Training** → Using transfer learning with ResNet18
3. **Evaluation** → Testing accuracy and interpreting results
4. **Deployment** → Creating a web app for real-world use

---

## 🗂️ Repository Structure

```
Food-Freshness-Detector/
├── app.py                    # Gradio web interface for inference
├── train.ipynb               # Complete training workflow (27 cells)
├── requirements.txt          # Python dependencies (pinned versions)
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
├── .gitignore                # Excluded files for git
├── DEPLOYMENT_GUIDE.md       # Step-by-step deployment instructions
├── PROJECT_WALKTHROUGH.md    # This file - explains how everything works
│
├── data/                     # Training dataset
│   ├── README.md             # Dataset instructions
│   └── food_freshness/       # Images organized by label
│       ├── fresh/            # 24 synthetic green ellipse images
│       ├── slightly_spoiled/ # 24 synthetic yellow ellipse images
│       └── rotten/           # 24 synthetic brown ellipse images
│
├── model/                    # Trained model artifacts
│   └── export.pkl            # Exported FastAI learner (45MB)
│
├── assets/                   # Demo resources
│   └── demo.png              # 256x256 sample image for testing
│
└── huggingface/              # Deployment configuration
    └── README.md             # Hugging Face Space card template
```

---

## 🔄 Data Flow: From Image to Prediction

### 1. Training Pipeline (`train.ipynb`)

```
Raw Images (data/food_freshness/)
    ↓
DataBlock: Define how to process images
    - get_items=get_image_files → Find all PNG files
    - get_y=parent_label → Use folder name as label
    - splitter=RandomSplitter → 80% train, 20% validation
    - item_tfms=Resize(224) → Resize to 224x224 pixels
    ↓
DataLoaders: Create batches for training
    - Batch size: 16 images
    - Vocabulary: ['fresh', 'rotten', 'slightly_spoiled']
    ↓
vision_learner: Load pretrained ResNet18
    - Pretrained on ImageNet (1.2M images)
    - Replace final layer for 3-class output
    - Freeze body layers, train head only (1 epoch)
    ↓
fine_tune: Unfreeze all layers
    - Train for 3 more epochs
    - Use discriminative learning rates
    - Lower LR for early layers, higher for later layers
    ↓
Export: Save complete pipeline
    - Saves DataBlock transforms
    - Saves model weights
    - Saves vocabulary
    - Creates model/export.pkl (45MB)
```

**Key FastAI Concepts Used:**
- **DataBlock API**: Declarative way to define data processing
- **Transfer Learning**: Reuse ResNet18 knowledge, adapt to food classification
- **Fine-Tuning**: Two-stage training (frozen → unfrozen)
- **Learner Export**: Bundles everything needed for inference

---

### 2. Inference Pipeline (`app.py`)

```
User Uploads Image (via Gradio web UI)
    ↓
predict_freshness(image: PIL.Image)
    - Save image to temporary file
    - Load via PILImage.create(temp_path) → FastAI format
    ↓
learner.predict(fastai_image)
    - Apply same transforms (Resize 224x224)
    - Pass through ResNet18 model
    - Get softmax probabilities for 3 classes
    ↓
Return Results
    - pred_class: 'fresh', 'slightly_spoiled', or 'rotten'
    - confidence: {class: probability} for all 3 classes
    - description: User-friendly text explanation
    ↓
Gradio Displays
    - Prediction label (e.g., "fresh")
    - Confidence scores (e.g., fresh: 85%, rotten: 12%, ...)
    - Description (e.g., "Looks fresh and ready to eat.")
```

**Why Temp File Approach?**
- Gradio passes `PIL.Image` objects directly
- FastAI's `PILImage.create()` works best with file paths
- Saving to temp file ensures compatibility with FastAI's transform pipeline

---

## 📄 File-by-File Explanation

### Core Application Files

#### `app.py` (55 lines)
**Purpose**: Web interface for real-time predictions  
**Key Components**:
- `load_learner(MODEL_PATH)`: Loads exported model at startup
- `predict_freshness(image)`: Core inference function
  - Saves PIL Image to temp file
  - Calls `learner.predict(fastai_image)`
  - Returns class, confidence dict, description
- Gradio Blocks UI:
  - `gr.Image(type="pil")`: Upload widget
  - `gr.Label()`: Shows confidence scores as bar chart
  - `gr.Textbox()`: Displays description text
  - `gr.Examples()`: Pre-loaded demo images

**Dependencies**: `fastai`, `gradio==4.20.0`, `huggingface-hub==0.20.0`

---

#### `train.ipynb` (27 cells)
**Purpose**: Complete training workflow with teaching commentary  
**Structure**:
1. **Introduction** (Markdown): Project overview and goals
2. **Setup & Imports** (Code): Import FastAI, verify GPU/MPS
3. **Dataset Overview** (Markdown + Code): Explain folder structure, show sample images
4. **DataBlock Definition** (Code + Markdown): Teach DataBlock API concepts
5. **Data Exploration** (Code): Visualize batches, check class distribution
6. **Model Creation** (Code): Create `vision_learner` with ResNet18
7. **Learning Rate Finder** (Code): Run `lr_find()`, plot results
8. **Training** (Code): `fine_tune(3, base_lr=1e-3)`
9. **Evaluation** (Code): `ClassificationInterpretation`, confusion matrix
10. **Export** (Code): Save model as `model/export.pkl`
11. **Testing** (Code): Load exported model, test on validation image
12. **Summary** (Markdown): Recap concepts learned

**Teaching Focus**: Each cell explains *why* we're doing something, not just *how*

---

### Configuration Files

#### `requirements.txt`
**Purpose**: Pin exact package versions for reproducibility  
**Critical Pins**:
- `gradio==4.20.0` (not 4.44+, which has HfFolder import issues)
- `huggingface-hub==0.20.0` (compatible with gradio 4.20.0)
- `fastai>=2.7.16` (latest stable release)

**Why Pin Versions?**  
Gradio/HuggingFace Hub have breaking changes between versions. Pinning ensures deployment works.

---

#### `.gitignore`
**Purpose**: Exclude files from version control  
**Key Exclusions**:
- `.venv/` - Virtual environment (regenerate locally)
- `__pycache__/` - Python bytecode
- `.ipynb_checkpoints/` - Jupyter autosave files
- `.DS_Store` - macOS metadata

**Included in Git**:
- `model/export.pkl` - Model file (45MB, required for deployment)
- `data/food_freshness/` - Training images (needed to re-train)

---

### Documentation Files

#### `README.md`
**Purpose**: Main project documentation for GitHub/portfolio  
**Sections**:
- Features & Architecture
- Dataset description
- Installation instructions
- Training guide
- Local testing
- Deployment steps
- Results & future improvements

**Audience**: Recruiters, collaborators, other developers

---

#### `DEPLOYMENT_GUIDE.md`
**Purpose**: Step-by-step deployment walkthrough  
**Content**:
- Exact git commands to push to Hugging Face
- Troubleshooting common build errors
- Files checklist before deployment
- Testing steps

**Audience**: You (when deploying updates) or anyone forking the project

---

#### `PROJECT_WALKTHROUGH.md` (this file)
**Purpose**: Explain how all pieces connect  
**Content**:
- Repository structure breakdown
- Data flow diagrams (training + inference)
- File-by-file purpose explanations
- Conceptual connections to FastAI book chapters

**Audience**: Your future self, learning portfolio reviewers

---

### Model & Data Files

#### `model/export.pkl` (45MB)
**Purpose**: Complete trained model ready for inference  
**Contents**:
- DataBlock transforms (Resize to 224x224)
- ResNet18 architecture + trained weights
- Vocabulary: `['fresh', 'rotten', 'slightly_spoiled']`
- Normalization stats (ImageNet mean/std)

**How It Was Created**:
```python
learn = vision_learner(dls, resnet18, metrics=[accuracy, error_rate])
learn.fine_tune(3, base_lr=1e-3)
learn.export('model/export.pkl')  # Saves everything
```

**How It's Loaded**:
```python
learner = load_learner('model/export.pkl')  # One line to restore
learner.predict(image)  # Ready to use
```

---

#### `data/food_freshness/` (72 images)
**Purpose**: Training dataset for model  
**Current State**: Synthetic images (color-coded ellipses)  
**Real-World Use**: Replace with actual food photos for production

**Folder Organization**:
```
data/food_freshness/
├── fresh/            # Green ellipses (represents healthy food)
│   ├── img_001.png
│   ├── img_002.png
│   └── ... (24 total)
├── slightly_spoiled/ # Yellow ellipses (mild spoilage)
│   ├── img_001.png
│   └── ... (24 total)
└── rotten/           # Brown ellipses (severe spoilage)
    ├── img_001.png
    └── ... (24 total)
```

**Why Folder Structure Matters**:
- FastAI's `parent_label` uses folder name as label
- No manual CSV or JSON labeling needed
- Easy to add new classes (just create new folder)

---

## 🧠 Conceptual Flow: FastAI Chapters 1-4 Concepts

### Chapter 1: Your First Model
✅ **Concept**: Train a classifier with minimal code  
🔗 **Used in Project**: `vision_learner(dls, resnet18).fine_tune(3)`

### Chapter 2: From Model to Production
✅ **Concept**: Export model and deploy to web app  
🔗 **Used in Project**: `learn.export()` + Gradio app + HF Spaces

### Chapter 3: Data Ethics
✅ **Concept**: Understand biases in training data  
🔗 **Used in Project**: Synthetic data disclaimer, real-world data recommendations

### Chapter 4: Under the Hood
✅ **Concept**: DataBlock, transforms, transfer learning mechanics  
🔗 **Used in Project**:
- DataBlock API: `ImageBlock`, `CategoryBlock`, `get_y=parent_label`
- Transforms: `Resize(224)`
- Transfer Learning: ResNet18 pretrained weights
- Fine-Tuning: Two-stage training (frozen → unfrozen)

---

## 🎓 Learning Outcomes

By building this project, you've demonstrated:

1. **Data Engineering**: Organized images into label-based folders
2. **Model Training**: Used transfer learning with ResNet18
3. **Evaluation**: Checked accuracy, confusion matrix, interpretation
4. **Deployment**: Created Gradio app, deployed to Hugging Face Spaces
5. **Documentation**: Wrote README, deployment guide, walkthrough
6. **Version Control**: Used git to track changes and deploy

**Interviewer Questions You Can Answer**:
- "How does transfer learning work?" → Pretrained ResNet18, fine-tune for food
- "What's in your export.pkl file?" → Transforms, model weights, vocabulary
- "Why Gradio?" → Fast prototyping, shareable demos, HF Spaces integration
- "How would you improve this?" → Real dataset, Grad-CAM, confidence thresholds

---

## 🔗 How Files Interact

```
train.ipynb → model/export.pkl → app.py
     ↓                              ↓
data/food_freshness/          Gradio Web UI
     ↓                              ↓
requirements.txt          Hugging Face Spaces
     ↓                              ↓
.gitignore → git → HF Spaces Deployment
```

**Training Phase**:
1. `train.ipynb` reads images from `data/food_freshness/`
2. Creates DataLoaders, trains ResNet18
3. Exports to `model/export.pkl`

**Deployment Phase**:
1. `app.py` loads `model/export.pkl`
2. Gradio creates web interface
3. User uploads image → `predict_freshness()` → Display results

**Version Control**:
1. `.gitignore` excludes temporary files
2. `requirements.txt` ensures dependency consistency
3. Git pushes to Hugging Face Spaces
4. HF automatically builds app from `app.py` + `requirements.txt`

---

## 🚀 Ready to Deploy?

1. Read `DEPLOYMENT_GUIDE.md` for exact commands
2. Create Hugging Face Space
3. Run git push
4. Share your live demo link!

**Your project is complete and production-ready.** 🎉
