from node import Node, Edge

import logging
from typing import List, Tuple, Any

import numpy as np
import torch
import Goban


class MCTS:

    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Max Game duration
        self.T: int = 200
        self.v_resign: float = -0.99
        self.simulations_per_play: int = 100
        # Number of moves after which temperature is set to something close to 0
        self.moves_after_low_temperature: int = 7  # defined empirically/proportionally
        # Number of boards to keep for history
        self.len_history: int = 7
        self.tensor_size: int = 2 * self.len_history + 1
        self.logger: logging.Logger = logging.getLogger("MCTS")
        self.board: Goban.Board = Goban.Board()
        self.torch_board = torch.from_numpy(self.board._board)

    def get_state(self):
        return self._vector, self.export_save()

    def is_closed(self) -> bool:
        return self.board.is_game_over()

    def set_game(self, state):
        """
        State is a torch tensor, like _vector.
        """
        self._vector, save = state
        self.load_save(save)

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
            state[0, :, :] = torch.reshape(self.torch_board == 0, (9, 9))
            state[1, :, :] = torch.reshape(self.torch_board == 1, (9, 9))
            # Swap turn
            state[self.tensor_size - 1] = 1 - self._vector[self.tensor_size - 1]
            states.append((state, self.export_save()))
            self.board.pop()
        return actions, states

    def evaluate(self, state) -> Tuple[np.ndarray, float]:
        """
        Evaluate the specified state, which is a torch tensor.
        """
        state = state.view(1, *state.shape).float()

        p, v = self.model(state)

        return p.detach().cpu().numpy()[0], v.detach().cpu().numpy()[0][0]

    def reset(self):
        """
        Reset and init the current game.
        """
        self.board: Goban.Board = Goban.Board()
        self._vector = torch.zeros((self.tensor_size, 9, 9), dtype=int)

    def get_winner(self) -> int:
        # TODO: Replace by Taylor-Tromp score
        (black, white) = self.board.compute_score()
        if black > white:
            return 0
        else:
            return 1

    def self_play(self) -> List[List[Any]]:
        """
        Does one game of selfplay.
        """
        self.reset()
        training_data = []
        coeff: int = 1
        root: Node = Node(self.get_state(), None)
        played_turns: int = 0
        temperature: float = 1.0
        resigned: bool = False
        while played_turns < self.T and not self.is_closed():
            if root.should_resign(self.v_resign):
                self.logger.debug(f"Resigned at turn {played_turns}")
                resigned = True
                break
            self.logger.log(7, f"Turn {played_turns} start")
            root.add_dirichlet_noise()
            for _ in range(self.simulations_per_play):
                node: Node = root.select()  # TODO: choose parameter cpuct
                self.set_game(node.state)
                if self.is_closed():
                    node.backup(node.inbound.current_action_value)
                else:
                    actions, states = self.explore_legal_moves()
                    priors, value = self.evaluate(node.state[0])
                    node.expand(actions, states, priors, value)
            self.logger.log(5, "Simulations completed")
            play_tuple = root.play(temperature)
            edge: Edge = play_tuple[0]
            pi: np.ndarray = play_tuple[1]
            self.set_game(edge.child.state)
            # Save training data
            training_data.append([root.state[0], pi, coeff])
            # Change root and free it
            root.free_except(edge.child)

            root: Node = edge.child
            self.logger.log(7, f"Turn {played_turns} terminated")

            coeff *= -1
            played_turns += 1

            if played_turns == self.moves_after_low_temperature:
                temperature: float = 0
                self.logger.log(7, "Temperature is now low")
        self.logger.log(8, "Game is finished")
        # Free the tree
        root.free_except(None)
        if resigned:
            # Last player had coeff and he resigned
            reward: int = -coeff
        else:
            reward: int = 1 if self.get_winner() == 0 else -1
        # backprop reward
        for triplet in training_data:
            triplet[2] *= reward
        return training_data

    def export_save(self):
        self.board._pushBoard()
        save = self.board._trailMoves.pop()
        save.append(self.board._historyMoveNames.copy())
        return save

    def load_save(self, save: List):
        self.board._historyMoveNames = save[-1].copy()
        self.board._currentHash = save[-2]
        self.board._empties = save[-3].copy()
        self.board._stringSizes = save[-4].copy()
        self.board._stringLiberties = save[-5].copy()
        self.board._stringUnionFind = save[-6].copy()
        self.board._lastPlayerHasPassed = save[-7]
        self.board._gameOver = save[-8]
        self.board._board = save[-9].copy()
        self.board._nextPlayer = save[-10]
        self.board._capturedBLACK = save[-11]
        self.board._capturedWHITE = save[-12]
        self.board._nbBLACK = save[-13]
        self.board._nbWHITE = save[-14]
