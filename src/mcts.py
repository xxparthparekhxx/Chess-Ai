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
    def __init__(self, model, num_simulations=800, c_puct=1.0, batch_size=8, virtual_loss=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = max(1, batch_size)
        self.virtual_loss = virtual_loss

    def search(self, board, temperature=0.0):
        # Create root
        root = Node(prior=0)
        
        # Expand root first to get legal moves
        policy, value = self.evaluate(board)
        self.expand(root, board, policy)
        
        # Add exploration noise to root (dirichlet)
        self.add_dirichlet_noise(root)

        remaining = self.num_simulations
        while remaining > 0:
            batch_size = min(self.batch_size, remaining)
            leaves = []

            # 1. Select a batch of leaves. Virtual loss makes the paths already
            #    selected in this batch less attractive to the next selection.
            for _ in range(batch_size):
                node = root
                search_board = board.copy()
                path = [node]

                while node.is_expanded():
                    action, node = self.select_child(node)
                    path.append(node)
                    move = chess.Move.from_uci(INT_TO_MOVE[action])
                    search_board.push(move)

                self.apply_virtual_loss(path)
                leaves.append((node, path, search_board))

            # 2. Evaluate all non-terminal leaves in a single network call.
            pending = []
            boards_to_evaluate = []
            for node, path, search_board in leaves:
                value, is_terminal = self.get_game_result(search_board)
                if is_terminal:
                    pending.append((node, path, search_board, value))
                else:
                    pending.append((node, path, search_board, None))
                    boards_to_evaluate.append(search_board)

            policies, values = self.evaluate_batch(boards_to_evaluate)
            evaluated = iter(zip(policies, values))

            # 3. Expand and backpropagate. Virtual loss is removed first so the
            #    real visit counts reflect exactly one simulation per leaf.
            for node, path, search_board, terminal_value in pending:
                if terminal_value is None:
                    policy, value = next(evaluated)
                    # A forced line can select the same leaf more than once in
                    # one batch; only the first visit should expand it.
                    if not node.is_expanded():
                        self.expand(node, search_board, policy)
                else:
                    value = terminal_value

                self.undo_virtual_loss(path)
                self.backpropagate(path, value, search_board.turn)

            remaining -= batch_size

        return self.select_best_move(root, board, temperature=temperature)

    def evaluate(self, board):
        policies, values = self.evaluate_batch([board])
        return policies[0], values[0]

    def evaluate_batch(self, boards):
        if not boards:
            return [], []

        tensors = np.stack([board_to_tensor(b) for b in boards])
        tensor = torch.from_numpy(tensors).to(DEVICE)
        with torch.no_grad():
            policy_logits, value = self.model(tensor)

        policies = F.softmax(policy_logits, dim=1).cpu().numpy()
        values = value.view(-1).cpu().numpy()

        # Helper: Relative Value
        # Model predicts +1 (White Win), -1 (Black Win)
        # We need to return value from perspective of current player
        # If Black to play is evaluated as +0.9 (White has advantage),
        # relative value for Black is -0.9.
        relative_values = [
            float(v) if board.turn == chess.WHITE else -float(v)
            for v, board in zip(values, boards)
        ]

        return policies, relative_values

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

    def apply_virtual_loss(self, path):
        # Node values are stored from the perspective of the player to move at
        # that node, so adding to a child's value_sum lowers its Q from the
        # parent's perspective. Applying it below the root penalizes every edge
        # on the path while the batch is being collected.
        for node in path[1:]:
            node.visit_count += 1
            node.value_sum += self.virtual_loss

    def undo_virtual_loss(self, path):
        for node in path[1:]:
            node.visit_count -= 1
            node.value_sum -= self.virtual_loss

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

    def select_best_move(self, root, board, temperature=0.0):
        if not root.children:
            return None, 0.0 # Should not happen unless no legal moves

        actions = list(root.children.keys())
        visit_counts = np.array(
            [root.children[action].visit_count for action in actions],
            dtype=np.float64,
        )

        if temperature <= 0.0:
            # Greedily select most visited
            best_action = actions[int(np.argmax(visit_counts))]
        else:
            # Sample from visit_count ** (1 / temperature).
            # Dividing by the max count leaves the distribution unchanged and
            # avoids overflow when temperature is small.
            max_count = visit_counts.max()
            if max_count <= 0:
                probs = np.full(len(actions), 1.0 / len(actions))
            else:
                weights = (visit_counts / max_count) ** (1.0 / temperature)
                probs = weights / weights.sum()
            best_action = actions[int(np.random.choice(len(actions), p=probs))]

        uci = INT_TO_MOVE[best_action]
        # Return predicted value (Q of best child)
        return uci, -root.children[best_action].value() 
