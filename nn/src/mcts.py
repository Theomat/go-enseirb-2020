from node import Node, Edge

from typing import List, Tuple

import numpy as np
import torch
import Goban


class MCTS:

    def __init__(self, model):
        self.model = model
        # Max Game duration
        self.T: int = 200
        self.v_resign: float = -99
        self.simulations_per_play: int = 1600
        # Number of moves after which temperature is set to something close to 0
        self.moves_after_low_temperature: int = 7  # defined empirically/proportionally
        # Number of boards to keep for history
        self.len_history: int = 7

        self.tensor_size: int = 2 * self.len_history + 1

    def get_state(self):
        return self._vector, self.board.export_save()

    def is_closed(self) -> bool:
        return self.board.is_game_over()

    def set_game(self, state):
        """
        State is a torch tensor, like _vector.
        """
        self._vector, save = state
        self.board.load_save(save)

    def explore_legal_moves(self) -> Tuple[List[int], List]:
        actions: List[int] = []
        states = []
        for action in self.board.weak_legal_moves():
            success = self.board.push(action)
            if not success:
                self.board.pop()
                continue
            actions.append(action)
            state = torch.zeros_like(self._vector)
            # Rollout the history
            state[2:self.tensor_size - 1, :, :] = self._vector[:self.tensor_size - 3, :, :]
            # Add board features
            state[0, :, :] = np.reshape(self.board == 0, (9, 9))
            state[1, :, :] = np.reshape(self.board == 1, (9, 9))
            # Swap turn
            state[self.tensor_size - 1] = 1 - self._vector[self.tensor_size - 1]
            states.append((state, self.board.export_save()))
            self.board.pop()
        return actions, states

    def evaluate(self, state) -> Tuple[np.ndarray, float]:
        """
        Evaluate the specified state, which is a torch tensor.
        """
        output: np.ndarray = self.model(state).detach().cpu().numpy()
        # TODO: adapt to torch because I belive the output isn't shaped like that
        return output[0], output[1:]

    def reset(self):
        """
        Reset and init the current game.
        """
        self.board: Goban.Board = Goban.Board()
        self._vector = torch.zeros((self.tensor_size, 9, 9), dtype=int)

    def self_play(self) -> List[List[np.ndarray, np.ndarray, int]]:
        """
        Does one game of selfplay.
        """
        self.reset()
        training_data = []
        coeff: int = 1
        root: Node = Node(self.get_state(), None)
        played_turns: int = 0
        temperature: float = 1.0
        while played_turns < self.T and not self.is_closed() and not root.should_resign(self.v_resign):
            root.add_dirichlet_noise()
            for _ in range(self.simulations_per_play):
                node: Node = root.select()  # TODO: choose parameter cpuct
                self.set_game(node.state)
                actions, states = self.explore_legal_moves()
                priors, value = self.evaluate(node.state[0])
                node.expand(actions, states, priors, value)

            tuple = root.play(temperature)
            edge: Edge = tuple[0]
            pi: np.ndarray = tuple[1]
            self.set_game(edge.child.state)
            # Save training data
            training_data.append([root.state[0], pi, coeff])
            # Change root and free it
            root.free_except(edge.child)
            root = edge.child

            coeff *= -1
            played_turns += 1

            if played_turns > self.moves_after_low_temperature:
                temperature = 10**-5  # defined empirically

        reward: int = 1 if self.get_winner() == 0 else -1
        # backprop reward
        for triplet in training_data:
            triplet[2] *= reward
        return training_data
