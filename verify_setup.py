import sys
import os
sys.path.append(os.getcwd())

try:
    from ChessAI.src.dataset import board_to_tensor, MOVE_TO_INT
    from ChessAI.src.model import ChessNet
    import chess
    import torch

    print("Successfully imported modules.")

    # Test Board Tensor
    board = chess.Board()
    tensor = board_to_tensor(board)
    print(f"Tensor shape: {tensor.shape}")
    assert tensor.shape == (14, 8, 8), f"Expected (14, 8, 8), got {tensor.shape}"

    # Test Model
    model = ChessNet()
    policy, value = model(torch.randn(1, 14, 8, 8))
    print(f"Policy shape: {policy.shape}, Value shape: {value.shape}")
    
    print("Verification Passed!")
except Exception as e:
    print(f"Verification Failed: {e}")
    import traceback
    traceback.print_exc()
