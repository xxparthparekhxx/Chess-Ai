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

# Load Model
print(f"Loading model on {DEVICE}...")
model = ChessNet(num_res_blocks=NUM_RES_BLOCKS).to(DEVICE)
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model (might be saving right now?): {e}")
else:
    print("Warning: No model found. Using random weights.")
model.eval()

def get_ai_move_logic(board):
    # 1. Prepare Input
    tensor = board_to_tensor(board)
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # 2. Model Prediction
    with torch.no_grad():
        policy_logits, value = model(tensor)
    
    # 3. Create Mask
    legal_moves = list(board.legal_moves)
    legal_indices = []
    for move in legal_moves:
        uci = move.uci()
        if uci in MOVE_TO_INT:
            legal_indices.append(MOVE_TO_INT[uci])
            
    if not legal_indices:
        return None, 0.0
    
    # 4. Apply Mask
    mask = torch.full_like(policy_logits, -float('inf'))
    mask[0, legal_indices] = 0
    masked_logits = policy_logits + mask
    
    # 5. Select Move
    probs = F.softmax(masked_logits, dim=1)
    move_index = torch.argmax(probs).item()
    move_uci = INT_TO_MOVE[move_index]
    
    # Value output (for display)
    val = value.item()
    
    return move_uci, val

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
