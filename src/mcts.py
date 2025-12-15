import math
import numpy as np
import torch
import torch.nn.functional as F
import chess
from dataset import board_to_tensor, MOVE_TO_INT, INT_TO_MOVE
from config import DEVICE

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.children = {} # Map move_index -> Node

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, num_simulations=800, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, board):
        # Create root
        root = Node(prior=0)
        
        # Expand root first to get legal moves
        policy, value = self.evaluate(board)
        self.expand(root, board, policy)
        
        # Add exploration noise to root (dirichlet)
        self.add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root
            search_board = board.copy()
            path = [node]
            
            # 1. Select
            while node.is_expanded():
                action, node = self.select_child(node)
                path.append(node)
                move = chess.Move.from_uci(INT_TO_MOVE[action])
                search_board.push(move)

            # 2. Evaluate & Expand
            value, is_terminal = self.get_game_result(search_board)
            if not is_terminal:
                policy, v = self.evaluate(search_board)
                self.expand(node, search_board, policy)
                value = v
            else:
                # Terminal value is absolute (1 for White win, -1 for Black win)
                pass

            # 3. Backpropagate
            self.backpropagate(path, value, search_board.turn) 
            # Note: value is "Value of the board state for the current player"?
            # No, standard is: Value for White.
            # If search_board.turn is Black, it means White just moved.
            
        return self.select_best_move(root, board)

    def evaluate(self, board):
        tensor = board_to_tensor(board)
        tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            policy_logits, value = self.model(tensor)
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        
        # Helper: Relative Value
        # Model predicts +1 (White Win), -1 (Black Win)
        # We need to return value from perspective of current player
        # If Black to play is evaluated as +0.9 (White has advantage), 
        # relative value for Black is -0.9.
        val = value.item()
        if board.turn == chess.BLACK:
            val = -val
            
        return policy, val

    def select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            u = self.c_puct * child.prior * (math.sqrt(node.visit_count) / (1 + child.visit_count))
            # Q value: We typically view Q from perspective of player to move at Node
            # But we store "White Value".
            # If board at Node is White to play, we want max(Q).
            # If board at Node is Black to play, we want min(Q).
            # Simplified: AlphaZero uses "Value for player to move".
            # So Backprop flips sign.
            
            q = -child.value() # Invert value because child value is for Opponent 
             
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def expand(self, node, board, policy):
        legal_moves = list(board.legal_moves)
        policy_sum = 0
        for move in legal_moves:
            uci = move.uci()
            if uci in MOVE_TO_INT:
                idx = MOVE_TO_INT[uci]
                p = policy[idx]
                policy_sum += p
                node.children[idx] = Node(prior=p)
        
        # Normalize priors
        for child in node.children.values():
             child.prior /= policy_sum if policy_sum > 0 else 1

    def backpropagate(self, path, value, turn_at_leaf):
        # Value is from perspective of White (1=White Win, -1=Black Win).
        # OR Value is from perspective of Player to Move? 
        # Standard: Value is [-1, 1] relative to "current player".
        # Let's stick to AlphaZero: value is v for the player whose turn it is.
        # So when backpropping, we flip sign each step.
        
        # My implementation:
        # evaluate() returns tanh value (-1 to 1). This is "Score for current player".
        # get_game_result() returns 1 if Current Player wins? No.
        
        # Let's align:
        # leaf_value: Score for player at leaf node.
        # path[-1] is leaf node.
        
        current_value = value 
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += current_value
            current_value = -current_value # Flip for parent

    def get_game_result(self, board):
        if board.is_game_over():
            white_wins = 1 if board.result() == "1-0" else (-1 if board.result() == "0-1" else 0)
            # Return value from perspective of player to move (who is about to lose/win?)
            # If board.turn is White, and White wins (1), return 1.
            # If board.turn is Black, and White wins (1), return -1 (Black lost).
            turn_mult = 1 if board.turn == chess.WHITE else -1
            return white_wins * turn_mult, True
        return 0, False

    def add_dirichlet_noise(self, node):
        moves = list(node.children.keys())
        noise = np.random.dirichlet([0.3] * len(moves))
        for i, action in enumerate(moves):
            node.children[action].prior = 0.75 * node.children[action].prior + 0.25 * noise[i]

    def select_best_move(self, root, board):
        # Greedily select most visited
        best_count = -1
        best_action = None
        for action, child in root.children.items():
            if child.visit_count > best_count:
                best_count = child.visit_count
                best_action = action
        
        if best_action is None:
            return None, 0.0 # Should not happen unless no legal moves
            
        uci = INT_TO_MOVE[best_action]
        # Return predicted value (Q of best child)
        return uci, -root.children[best_action].value() 
