import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import os
import glob
import random
from model import ChessNet
from config import PROCESSED_DIR, BATCH_SIZE, LEARNING_RATE, EPOCHS, DEVICE, MODEL_PATH, NUM_RES_BLOCKS

class ChessIterableDataset(IterableDataset):
    def __init__(self, data_dir, shuffle_chunks=True):
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}. Run preprocess.py first.")
        self.shuffle_chunks = shuffle_chunks
        print(f"Found {len(self.files)} data chunks.")
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # If multi-worker, split files among workers
        if worker_info is None:
            files_to_read = self.files
        else:
            per_worker = int(np.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.files))
            files_to_read = self.files[start:end]
            
        if self.shuffle_chunks:
            random.shuffle(files_to_read)
            
        for f in files_to_read:
            # print(f"Loading {f}...") # Too verbose for training
            data = np.load(f)
            inputs = data['inputs']
            policies = data['policies']
            values = data['values']
            
            # Shuffle within the chunk
            indices = np.arange(len(inputs))
            if self.shuffle_chunks:
                np.random.shuffle(indices)
                
            for i in indices:
                yield (
                    torch.tensor(inputs[i], dtype=torch.float32),
                    torch.tensor(policies[i], dtype=torch.long),
                    torch.tensor(values[i], dtype=torch.float32)
                )

def train():
    try:
        # Use IterableDataset for lazy loading
        dataset = ChessIterableDataset(PROCESSED_DIR, shuffle_chunks=True)
    except FileNotFoundError as e:
        print(e)
        return

    # shuffle=False is required for IterableDataset
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Estimate total batches for tqdm
    # Each chunk has exactly 10,000 samples (from preprocess.py)
    total_samples = len(dataset.files) * 10000
    total_batches = total_samples // BATCH_SIZE
    
    model = ChessNet(num_res_blocks=NUM_RES_BLOCKS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loss functions
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    if os.path.exists(MODEL_PATH):
        print(f"Resuming form {MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
        except Exception:
            print("Could not load checkpoint, starting fresh.")

    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        batch_count = 0
        
        # TQDM Progress Bar
        progress_bar = tqdm(enumerate(dataloader), total=total_batches, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (inputs, target_policy, target_value) in progress_bar:
            inputs, target_policy, target_value = inputs.to(DEVICE), target_policy.to(DEVICE), target_value.to(DEVICE)
            
            optimizer.zero_grad()
            pred_policy, pred_value = model(inputs)
            
            loss_p = criterion_policy(pred_policy, target_policy)
            loss_v = criterion_value(pred_value.view(-1), target_value.view(-1))
            loss = loss_p + 0.5 * loss_v
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count = batch_idx
            
            # Update tqdm description with current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", p_loss=f"{loss_p.item():.4f}", v_loss=f"{loss_v.item():.4f}")
        
        avg_loss = total_loss / (batch_count + 1) if batch_count > 0 else 0
        print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
