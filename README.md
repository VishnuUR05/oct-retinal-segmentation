# Fundus Retinal Vessel Segmentation Module

## Purpose
Automated retinal blood vessel segmentation and image-derived vascular biomarker extraction.

## Main Features
- Fundus image upload
- Automatic retinal vessel segmentation
- Probability map
- Binary vessel mask
- Vessel overlay
- Skeleton visualization
- Vessel density
- Total vessel length
- Mean vessel width
- Median vessel width
- Tortuosity
- Branch points
- Endpoints
- Automatic ground-truth matching for supported FIVES validation images
- Dice, IoU, Precision, Recall, Specificity
- Pixel-wise error maps
- Biomarker deviation analysis
- Optional physical calibration support where valid

## Important Scientific Limitation
This module does NOT diagnose:
- Alzheimer's disease
- Dementia
- Any retinal disease

It performs vessel segmentation and extracts image-derived quantitative vascular features. Pixel values must not be converted to µm unless validated physical calibration information is available.

## Installation
1. Ensure Python 3.9+ is installed.
2. Install the necessary dependencies (typically `torch`, `torchvision`, `streamlit`, `opencv-python`, `scikit-image`, `scipy`, `pandas`).
3. You can create a virtual environment and install them:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt # (or manually install the packages listed above)
```

## Model Checkpoint
The trained segmentation model checkpoint (`best_model.pth`) exceeds GitHub file size limits (~280MB) and is **not** included in this repository.
Teammates must obtain `best_model.pth` separately and place it precisely here:
`fundus/vessel_module/checkpoints/best_model.pth`

## Running
Once the environment is active and the checkpoint is placed, run the Streamlit app:
```bash
cd 1.5fundus
streamlit run app.py
```

## Validation
Reproducible evaluation scripts are located in `1.5fundus/final_comprehensive_evaluation.py` and evaluation reports can be found in the `docs/` directory.
