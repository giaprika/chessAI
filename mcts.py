import numpy as np
import chess
import time
from collections import defaultdict
from utils import move_to_index

class MCTSNode:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior

    def is_expanded(self):
        return len(self.children) > 0

    def select_child(self):
        log_total = np.log(self.visit_count + 1)
        best_score = -np.inf
        best_move = None
        best_child = None

        for move, child in self.children.items():
            q_value = child.total_value / (child.visit_count + 1e-8)
            u_value = child.prior * np.sqrt(log_total) / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def expand(self, policy):
        legal_moves = list(self.board.legal_moves)
        total_prob = sum(policy[move_to_index(move)] for move in legal_moves)

        for move in legal_moves:
            prob = policy[move_to_index(move)] / (total_prob + 1e-8)
            next_board = self.board.copy()
            next_board.push(move)
            self.children[move] = MCTSNode(next_board, parent=self, prior=prob)

    def backpropagate(self, value):
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            self.parent.backpropagate(-value)

class MCTS:
    def __init__(self, model, time_limit):
        self.model = model
        self.time_limit = time_limit
        self.root = None

    def search(self, board):
        self.root = MCTSNode(board)
        policy, _ = self.model.predict(board)
        self.root.expand(policy)

        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            node = self.root
            path = [node]

            # Selection
            while node.is_expanded() and not node.board.is_game_over():
                move, node = node.select_child()
                path.append(node)

            # Evaluation
            if not node.board.is_game_over():
                policy, value = self.model.predict(node.board)
                node.expand(policy)
            else:
                result = node.board.result()
                if result == "1-0":
                    value = 1
                elif result == "0-1":
                    value = -1
                else:
                    value = 0

            # Backpropagation
            for node in reversed(path):
                node.backpropagate(value)
                value = -value

        best_move = max(self.root.children.items(), key=lambda x: x[1].visit_count)[0]
        return best_move
