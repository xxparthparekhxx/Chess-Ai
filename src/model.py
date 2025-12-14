import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import ACTION_SIZE

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=10):
        super().__init__()
        # Input: 14 channels (pieces + turn)
        self.conv_input = nn.Sequential(
            nn.Conv2d(14, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Tower of Residual Blocks
        self.res_tower = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_res_blocks)]
        )

        # Policy Head (Move Probabilities)
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, ACTION_SIZE) # Output: Logits for every possible move
        )

        # Value Head (Win/Loss Evaluation)
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() # Output: -1 (Loss) to 1 (Win)
        )

    def forward(self, x):
        x = self.conv_input(x)
        x = self.res_tower(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
