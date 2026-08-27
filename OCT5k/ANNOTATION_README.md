# OCT Retinal-Layer Annotation Tool

## Purpose
This application provides an interactive Streamlit UI to correct U-Net predicted retinal boundaries on external OCT images (OCT2017). The app automatically extracts the corrected 5 boundaries into CSV files and generates an integer 6-class segmentation mask suitable for training or analysis. 

Importantly, it **does not** retrain the model. It strictly acts as a human-in-the-loop annotation mechanism.

## Folder Structure
All outputs are strictly routed to `outputs/external_annotation/` to avoid modifying the original dataset.
- `selected_images.csv`: Deterministic manifest of the 80 images selected (20 per category).
- `metadata.csv`: Tracks annotation status (SAVED, SKIPPED) and timestamps.
- `[Category]/boundaries/`: The saved 512-column CSV files.
- `[Category]/masks/`: The saved integer 6-class `.png` segmentation masks.

## How to Launch
Activate the virtual environment and start Streamlit:

```powershell
.venv\Scripts\Activate.ps1
streamlit run annotation_app.py
```

## First-Phase Pilot
We recommend starting with a small 20-image pilot:
- 5 DME
- 5 CNV
- 5 Drusen
- 5 Normal

Use the "Filter Category" dropdown in the sidebar to switch between categories and stop after annotating the first 5 images of each.

## Annotation Procedure
1. **Load Image:** The model automatically runs a zero-shot inference pass.
2. **Review:** Look at the generated boundary lines.
3. **Select Boundary & Action:** Use the radio buttons to choose an active boundary (e.g., ILM) and an action (Add Point, Move Nearest, Delete Nearest).
4. **Edit:** Click directly on the OCT image canvas. The nearest control point will update, and the boundary will smoothly interpolate across the image.
5. **Mark Uncertain:** If a boundary completely degrades or is not visible, uncheck "Boundary is Valid / Visible". This sets it to NaN.
6. **Verify:** Check the verification panel at the bottom to ensure the 6-class mask was generated correctly and no "Anatomical Errors" (crossings) are flagged.
7. **Save:** Click the primary "SAVE ANNOTATION" button. (Warning: Next/Previous will not auto-save).

## Meaning of the Boundaries
1. **ILM**: Inner Limiting Membrane (Red)
2. **OPL**: Outer Plexiform Layer (Green)
3. **IS-OS**: Inner Segment / Outer Segment Junction (Blue)
4. **IBRPE**: Inner Boundary of Retinal Pigment Epithelium (Cyan)
5. **OBRPE**: Outer Boundary of Retinal Pigment Epithelium (Magenta)

## Meaning of the 6 Classes
- **Class 0:** Background above ILM
- **Class 1:** Between ILM and OPL
- **Class 2:** Between OPL and IS-OS
- **Class 3:** Between IS-OS and IBRPE
- **Class 4:** Between IBRPE and OBRPE
- **Class 5:** Background below OBRPE
