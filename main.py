import torch
import torch.nn.functional as F
import chess
import numpy as np
import sys
import os

from src.model import ChessNet
from src.dataset import board_to_tensor, MOVE_TO_INT, INT_TO_MOVE
from src.config import MODEL_PATH, DEVICE, NUM_RES_BLOCKS

def load_model():
    model = ChessNet(num_res_blocks=NUM_RES_BLOCKS).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("No trained model found! Initializing with random weights.")
    model.eval()
    return model

def get_ai_move(model, board):
    # 1. Prepare Input
    tensor = board_to_tensor(board)
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # 2. Model Prediction
    with torch.no_grad():
        policy_logits, value = model(tensor)
    
    # 3. Create Mask for Legal Moves
    legal_moves = list(board.legal_moves)
    legal_indices = []
    for move in legal_moves:
        uci = move.uci()
        if uci in MOVE_TO_INT:
            legal_indices.append(MOVE_TO_INT[uci])
            
    if not legal_indices:
        return None # No legal moves (Checkmate/Stalemate)
    
    # 4. Apply Mask
    # Initialize mask with -inf (log probability of 0)
    # We use a large negative number to effectively zero out probability after softmax
    mask = torch.full_like(policy_logits, -float('inf'))
    mask[0, legal_indices] = 0 # allowing original logits to pass through (logit + 0)
    
    masked_logits = policy_logits + mask
    
    # 5. Select Move (Greedy: Argmax, or Probabilistic: Multinomial)
    # For playing, typically use Argmax or minimal temperature
    probs = F.softmax(masked_logits, dim=1)
    move_index = torch.argmax(probs).item()
    
    # Convert back to UCI
    move_uci = INT_TO_MOVE[move_index]
    return chess.Move.from_uci(move_uci)

def main():
    model = load_model()
    board = chess.Board()
    
    print("\n--- ChessAI v1.0 ---\n")
    print("Enter moves in UCI format (e.g., e2e4). Type 'quit' to exit.")
    
    while not board.is_game_over():
        print(board)
        print(f"\nTurn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        if board.turn == chess.WHITE:
            # User Move
            user_move = input("Your Move: ").strip()
            if user_move.lower() == 'quit':
                break
            
            try:
                move = chess.Move.from_uci(user_move)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move! Try again.")
            except ValueError:
                print("Invalid format. Use UCI (e.g. e2e4)")
                
        else:
            # AI Move
            print("AI is thinking...")
            ai_move = get_ai_move(model, board)
            if ai_move:
                print(f"AI plays: {ai_move.uci()}")
                board.push(ai_move)
            else:
                print("AI has valid moves but mapping failed or game over.")
                break

    print("Game Over")
    print(board.result())

if __name__ == "__main__":
    main()
