import torch
import torch.nn.functional as F
import chess
import numpy as np
import sys
import os

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import ChessNet
from src.dataset import MOVE_TO_INT, INT_TO_MOVE
from src.config import MODEL_PATH, DEVICE, NUM_RES_BLOCKS
from src.mcts import MCTS

def load_model():
    model = ChessNet(num_res_blocks=NUM_RES_BLOCKS).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("No trained model found! Initializing with random weights.")
    model.eval()
    return model

def main():
    model = load_model()
    # Initialize MCTS with the model
    mcts = MCTS(model, num_simulations=800)
    
    board = chess.Board()
    
    print("\n--- ChessAI v1.1 (MCTS Enabled) ---\n")
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
            print("AI is thinking (MCTS Search)...")
            
            # Simple temperature schedule
            temp = 0.5 if len(board.move_stack) < 10 else 0.0
            
            # Use MCTS search
            best_move_uci, eval_value = mcts.search(board)
            
            if best_move_uci:
                print(f"AI plays: {best_move_uci} (Eval: {eval_value:.2f})")
                board.push(chess.Move.from_uci(best_move_uci))
            else:
                print("AI could not find a move. Game Over?")
                break

    print("Game Over")
    print(board.result())

if __name__ == "__main__":
    main()
