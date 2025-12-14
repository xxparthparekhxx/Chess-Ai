import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from model import ChessNet
from config import PROCESSED_DIR, BATCH_SIZE, LEARNING_RATE, EPOCHS, DEVICE, MODEL_PATH, NUM_RES_BLOCKS

class ChessDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}. Run preprocess.py first.")
        
        # Load all data into memory (simplest for now, though Memory Mapping is better for huge datasets)
        # For a "Lite" version, we can iterate files, but let's try to load what we can.
        # If dataset is huge, we should implement an IterableDataset or load lazily.
        # Here we assume it fits in RAM for simplicity or we load just one file per epoch (naive).
        # Better approach for scalable training:
        print(f"Found {len(self.files)} data chunks.")
        self.inputs = []
        self.policies = []
        self.values = []
        
        for f in self.files:
            print(f"Loading {f}...")
            data = np.load(f)
            self.inputs.append(data['inputs'])
            self.policies.append(data['policies'])
            self.values.append(data['values'])
            
        self.inputs = np.concatenate(self.inputs)
        self.policies = np.concatenate(self.policies)
        self.values = np.concatenate(self.values)
        
        print(f"Total samples: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.policies[idx], dtype=torch.long),
            torch.tensor(self.values[idx], dtype=torch.float32)
        )

def train():
    try:
        dataset = ChessDataset(PROCESSED_DIR)
    except FileNotFoundError as e:
        print(e)
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = ChessNet(num_res_blocks=NUM_RES_BLOCKS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Loss functions
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (inputs, target_policy, target_value) in enumerate(dataloader):
            inputs, target_policy, target_value = inputs.to(DEVICE), target_policy.to(DEVICE), target_value.to(DEVICE)
            
            optimizer.zero_grad()
            
            pred_policy, pred_value = model(inputs)
            
            # Policy Loss: CrossEntropy expects (N, C) and (N)
            loss_p = criterion_policy(pred_policy, target_policy)
            
            # Value Loss: MSE expects (N, 1) and (N) or (N, 1)
            # target_value is (N), reshape to (N, 1)
            loss_v = criterion_value(pred_value.view(-1), target_value.view(-1))
            
            loss = loss_p + 0.5 * loss_v
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f} (P: {loss_p.item():.4f}, V: {loss_v.item():.4f})")
        
        print(f"Epoch {epoch+1} Completed. Avg Loss: {total_loss / len(dataloader):.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
