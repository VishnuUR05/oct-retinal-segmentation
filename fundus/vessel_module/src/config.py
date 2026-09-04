import os
import torch

SEED = 42
IMAGE_SIZE = 512
NUM_CLASSES = 1
BATCH_SIZE = 2
NUM_WORKERS = 0  # Safe for Windows
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 50
PATIENCE = 10
THRESHOLD = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

import os
# BASE_DIR is two levels up from src/ (fundus/vessel_module)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "generated_patches")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATA_DIR, "train", "masks")
VAL_IMG_DIR = os.path.join(DATA_DIR, "validation", "images")
VAL_MASK_DIR = os.path.join(DATA_DIR, "validation", "masks")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")
