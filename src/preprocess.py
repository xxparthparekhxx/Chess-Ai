import chess.pgn
import numpy as np
import os
import glob
from dataset import board_to_tensor, MOVE_TO_INT
from config import DATA_DIR, PROCESSED_DIR

def process_pgn(pgn_file_path, batch_size=10000):
    print(f"Processing {pgn_file_path}...")
    pgn = open(pgn_file_path, encoding='utf-8')
    
    inputs = []
    policy_targets = []
    value_targets = []
    
    game_count = 0
    batch_index = 0

    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        
        game_count += 1
        board = game.board()
        result = game.headers["Result"]
        
        # Parse result
        if result == "1-0":
            val = 1.0
        elif result == "0-1":
            val = -1.0
        else:
            val = 0.0 # Draw
            
        for move in game.mainline_moves():
            # 1. Input Tensor
            tensor = board_to_tensor(board)
            inputs.append(tensor)
            
            # 2. Policy Target (Move Index)
            uci = move.uci()
            if uci in MOVE_TO_INT:
                policy_targets.append(MOVE_TO_INT[uci])
            else:
                # Should not happen if map covers all, but for safety in "approx" map
                # Skip this position if move is unknown (e.g. some weird promotion?)
                inputs.pop() 
                board.push(move)
                continue

            # 3. Value Target (Game Result)
            # Standard AlphaZero uses the final game result for every position
            value_targets.append(val)
            
            board.push(move)
            
            # Save batch
            if len(inputs) >= batch_size:
                save_batch(inputs, policy_targets, value_targets, batch_index)
                inputs, policy_targets, value_targets = [], [], []
                batch_index += 1
                
        if game_count % 100 == 0:
            print(f"Processed {game_count} games...")

    # Save remaining
    if len(inputs) > 0:
        save_batch(inputs, policy_targets, value_targets, batch_index)
        
    print(f"Done. Processed {game_count} games.")

def save_batch(inputs, policies, values, index):
    filename = os.path.join(PROCESSED_DIR, f"chunk_{index}.npz")
    np.savez_compressed(
        filename,
        inputs=np.array(inputs, dtype=np.float32),
        policies=np.array(policies, dtype=np.int64),
        values=np.array(values, dtype=np.float32)
    )
    print(f"Saved {filename}")

if __name__ == "__main__":
    # Look for any .pgn file in data/
    pgn_files = glob.glob(os.path.join(DATA_DIR, "*.pgn"))
    if not pgn_files:
        print(f"No PGN files found in {DATA_DIR}")
    else:
        for f in pgn_files:
            process_pgn(f)
