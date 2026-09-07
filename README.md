# Retinal Vessel Segmentation & Multi-Model Biomarker Suite

Welcome to the Retinal Vessel AI Pipeline & Analysis Suite. This repository houses two complementary deep learning models designed for fundus retinal vessel segmentation and quantitative vascular biomarker extraction.

---

## 🔬 Model Comparison Overview

| Feature | Model 1: Custom U-Net (4-Stage) | Model 2: ResNet34-UNet |
| :--- | :--- | :--- |
| **Architecture** | Custom 4-Stage Encoder-Decoder (31M params) | Pretrained ResNet-34 Encoder + U-Net Decoder (24M params) |
| **Input Resolution** | 512 × 512 (RGB) | 512 × 512 (RGB) |
| **Primary Focus** | Vessel segmentation, vascular density heatmaps, 4-quadrant anatomical distribution | High-precision vessel segmentation, skeleton centerline extraction, tortuosity, branching points |
| **Key Metrics** | Global vessel density %, 4-quadrant density (SN, ST, IN, IT) | Vessel density %, centerline length, mean width, tortuosity index, branch points, endpoints |
| **Weights Path** | `outputs/fives_pipeline/best_fives_unet.pth` | `fundus/vessel_module/checkpoints/best_model.pth` |

---

## 🚀 Quick Start Guide (For Collaborators)

Follow these steps to pull the code, install dependencies, place model weights, and launch the application without any deviations.

### 1. Get the Code (Git / Pull Request)

#### If you are cloning for the first time:
```bash
git clone https://github.com/VishnuUR05/oct-retinal-segmentation.git
cd oct-retinal-segmentation
git checkout feature/fundus-pipeline
```

#### If you already have the repository cloned:
```bash
git fetch origin
git checkout feature/fundus-pipeline
git pull origin feature/fundus-pipeline
```

---

### 2. Environment Setup & Dependencies

It is recommended to use a dedicated Python 3.10+ virtual environment or conda environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip and install all required packages
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional for NVIDIA GPU acceleration)*:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### 3. Model Checkpoints Setup (Google Drive)

Download the trained model weights from the provided Google Drive link and place them in the exact directory paths below:

1. **`best_fives_unet.pth`** (~124 MB)
   - Destination: `outputs/fives_pipeline/best_fives_unet.pth`
2. **`best_model.pth`** (~45 MB / 280 MB)
   - Destination: `fundus/vessel_module/checkpoints/best_model.pth`

#### Directory Verification:
Make sure your folder structure looks like this:
```text
oct-retinal-segmentation/
├── fundus/
│   └── vessel_module/
│       └── checkpoints/
│           └── best_model.pth             <--- Place Model 2 weights here
├── outputs/
│   └── fives_pipeline/
│       └── best_fives_unet.pth            <--- Place Model 1 weights here
├── src/
│   └── fives_pipeline/
├── requirements.txt
└── streamlit_app.py
```
*(If the directories do not exist, create them manually or via terminal: `mkdir -p outputs/fives_pipeline` and `mkdir -p fundus/vessel_module/checkpoints`)*.

---

### 4. Dataset Setup (FIVES Dataset - WhatsApp Link)

If you are retraining or running validation scripts, download the FIVES dataset from the shared WhatsApp link and extract it:

```text
oct-retinal-segmentation/
└── data/
    └── fives/
        ├── train/
        │   ├── Original/
        │   └── Ground_truth/
        └── test/
            ├── Original/
            └── Ground_truth/
```
*(Note: To test the Streamlit web demo, you can simply upload any fundus JPG/PNG image directly via the browser interface without needing the entire dataset locally).*

---

### 5. Running the Unified Web Demo

Launch the unified dual-model comparison application:

```bash
streamlit run streamlit_app.py
```

- The app will automatically open at `http://localhost:8501`.
- Both models are auto-loaded from their default paths.
- Upload any retinal fundus image and click **"Run Unified Dual-Model Analysis"**.
- The dashboard presents side-by-side masks, density maps, skeleton metrics, quadrant analysis, and downloadable CSV comparison reports.

---

## 🛠️ Advanced Usage (Training & Batch Evaluation)

### Run Standalone FIVES Evaluation
```bash
python src/fives_pipeline/evaluate.py --data_dir data/fives --checkpoint outputs/fives_pipeline/best_fives_unet.pth
```

### Train Custom U-Net on FIVES
```bash
python src/fives_pipeline/train.py --data_dir data/fives --epochs 12 --batch_size 4 --lr 0.0003
```

---

## ❓ Troubleshooting & FAQs

1. **`TypeError: ImageMixin.image() got an unexpected keyword argument 'use_container_width'`**:
   - Resolved. The codebase is fully compatible with Streamlit versions (`use_column_width=True` standard).
2. **`KeyError: 'conv1.weight'` or Checkpoint Load Error**:
   - The checkpoint loader in `streamlit_app.py` automatically detects whether checkpoints contain raw state dicts or wrapped state dicts (`checkpoint['model_state_dict']`).
3. **Missing folder / FileNotFoundError**:
   - Double check that the folder names match casing: `outputs/fives_pipeline/` and `fundus/vessel_module/checkpoints/`.
