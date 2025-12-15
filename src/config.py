import torch
import os

# Hyperparameters
BATCH_SIZE = 3072
LEARNING_RATE = 1e-3
EPOCHS = 10
NUM_RES_BLOCKS = 10

# Data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
PGN_FILE = os.path.join(DATA_DIR, 'games.pgn') # Default PGN file

# Model
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'chess_net.pth')

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
# Create dirs if not exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
