import chess
import numpy as np
import torch
from torch.utils.data import Dataset

# --- 1. THE MOVE MAPPER ---
def create_uci_labels():
    """
    Generates a dictionary mapping every possible legal move in chess 
    (in UCI format) to an integer. approx 1968 moves.
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    # Normal moves and captures
    for l1 in letters:
        for n1 in numbers:
            destinations = [(l, n) for l in letters for n in numbers]
            for l2, n2 in destinations:
                if l1 == l2 and n1 == n2: continue # Same square
                labels_array.append(f"{l1}{n1}{l2}{n2}")

    # Promotions
    for l1 in letters:
        for l2 in letters: # Promotions only happen on files, sometimes captures
            # White promotions (row 7 -> 8)
            for p in promoted_to:
                labels_array.append(f"{l1}7{l2}8{p}")
            # Black promotions (row 2 -> 1)
            for p in promoted_to:
                labels_array.append(f"{l1}2{l2}1{p}")

    # Create the map
    return {move: i for i, move in enumerate(sorted(list(set(labels_array))))}

MOVE_TO_INT = create_uci_labels()
INT_TO_MOVE = {v: k for k, v in MOVE_TO_INT.items()}
ACTION_SIZE = len(MOVE_TO_INT) # ~1968

# --- 2. THE INPUT ENCODER ---
def board_to_tensor(board):
    # 14 channels, 8x8 board
    tensor = np.zeros((14, 8, 8), dtype=np.float32)
    
    # Piece placement (Channels 0-11)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Shift by 6 if black
            channel = piece_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
            row, col = divmod(square, 8)
            tensor[channel, row, col] = 1.0

    # Turn indication (Channels 12-13)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    else:
        tensor[13, :, :] = 1.0

    return tensor
