import shutil
import os
import sys

# Add src to path to find config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_DIR, DATA_DIR

def zip_processed_data():
    output_filename = os.path.join(DATA_DIR, 'chess_data')
    print(f"Zipping {PROCESSED_DIR} to {output_filename}.zip...")
    
    # Check if dir exists
    if not os.path.exists(PROCESSED_DIR):
        print(f"Error: {PROCESSED_DIR} does not exist. Run preprocess.py first.")
        return

    # Create zip
    shutil.make_archive(output_filename, 'zip', PROCESSED_DIR)
    print(f"Successfully created {output_filename}.zip")
    print(f"Upload this file to your Google Drive to train on Colab.")

if __name__ == "__main__":
    zip_processed_data()
