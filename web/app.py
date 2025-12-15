import sys
import os
import torch
import torch.nn.functional as F
import chess
from flask import Flask, render_template, request, jsonify

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model import ChessNet
from dataset import board_to_tensor, MOVE_TO_INT, INT_TO_MOVE
from config import MODEL_PATH, DEVICE, NUM_RES_BLOCKS

app = Flask(__name__)

from mcts import MCTS


# Load Model (Global)
print(f"Loading model on {DEVICE}...")
model = ChessNet(num_res_blocks=NUM_RES_BLOCKS).to(DEVICE)
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Warning: No model found. Using random weights.")
model.eval()

# Global MCTS instance
mcts = MCTS(model, num_simulations=800)

def get_ai_move_logic(board):
    # Use MCTS to search for the best move
    # Temperature 0 for competitive play (greedy selection from MCTS stats)
    # But for opening/variety we might want temp > 0.
    # For now, let's use temp=0.5 for first 10 moves, then 0.
    temp = 0.5 if len(board.move_stack) < 10 else 0.0
    
    best_move_uci, eval_value = mcts.search(board)
    return best_move_uci, eval_value

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def make_move():
    data = request.json
    fen = data.get('fen')
    if not fen:
        return jsonify({'error': 'No FEN provided'}), 400
    
    board = chess.Board(fen)
    if board.is_game_over():
         return jsonify({'game_over': True, 'result': board.result()})
         
    ai_move, ai_eval = get_ai_move_logic(board)
    if ai_move:
        return jsonify({'move': ai_move, 'eval': ai_eval})
    else:
        return jsonify({'error': 'No valid moves found (Checkmate?)'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
