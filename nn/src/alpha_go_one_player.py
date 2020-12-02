# -*- coding: utf-8 -*-
from node import Node, Edge
from model import AlphaGoZero

import numpy as np

import torch
from typing import List, Tuple

import Goban
from playerInterface import PlayerInterface


class myPlayer(PlayerInterface):
    '''
    AlphaGoOne
    '''

    def __init__(self):
        self._mycolor = None

        self.model = AlphaGoZero()
        self.model.load_state_dict(torch.load("model.pt"))

        self.device = "cpu"
        self.model.eval()
        self.max_depth: int = 100
        self.simulations_per_play: int = 100
        # Number of boards to keep for history
        self.len_history: int = 7
        self.tensor_size: int = 2 * self.len_history + 1
        self.board: Goban.Board = Goban.Board()
        self.np_array: np.ndarray = np.zeros((9, 9), dtype=np.float)
        self.torch_board = torch.from_numpy(self.np_array)

    def getPlayerName(self):
        return "AlphaGoOne"

    def getPlayerMove(self):
        if self.board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"
        for _ in range(self.simulations_per_play):
            node: Node = self.root.select(max=self._mycolor == Goban.Board._BLACK)  # TODO: choose parameter cpuct
            self.set_game(node.state)
            parent_depth: int = 0
            if node.inbound:
                parent: Node = node.inbound.parent
                if parent:
                    parent_depth: int = parent.depth
            if self.is_closed():
                node.backup(1 if self.get_winner() else -1)
            elif self.root.depth - parent_depth >= self.max_depth:
                node.backup(node.inbound.current_action_value)
            else:
                actions, states = self.explore_legal_moves()
                priors, value = self.evaluate(node.state[0])
                node.expand(actions, states, priors, value)
        play_tuple: Tuple[Edge, torch.FloatTensor] = self.root.play(0)
        edge: Edge = play_tuple[0]
        action: int = edge.action
        self.set_game(edge.child.state)
        self.root.free_except(edge.child)
        self.root: Node = edge.child
        print("[AlphaGoOne] Played with depth:", self.root.depth + 1, "with score:",
              edge.current_action_value, "~", edge.incertitude)
        return Goban.Board.flat_to_name(action)

    def playOpponentMove(self, move):
        print("Opponent played ", move, "i.e. ", move)  # New here
        flat = Goban.Board.name_to_flat(move)
        if self.root.depth == 0:
            raise("Failed")
        else:
            edge: Edge = self.root.get_child_for_action(flat)
            self.set_game(edge.child.state)
            self.root.free_except(edge.child)
            self.root: Node = edge.child
            print("[AlphaGoOne] Opponent choose node with score:", edge.current_action_value, "~", edge.incertitude)

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)
        self.reset()
        self.root: Node = Node(self.get_state(), None)
        if self._mycolor == Goban.Board._WHITE:
            actions, states = self.explore_legal_moves()
            priors, value = self.evaluate(self.root.state[0])
            self.root.expand(actions, states, priors, value)

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

            self.np_array[:, :] = self.board._board.reshape((9, 9))
            actions.append(action)
            state = torch.zeros_like(self._vector)
            # Rollout the history
            state[2:self.tensor_size - 1, :, :] = self._vector[:self.tensor_size - 3, :, :].clone().detach()
            # Add board features
            state[0, :, :] = torch.reshape(self.torch_board == 1, (9, 9)).clone().detach()
            state[1, :, :] = torch.reshape(self.torch_board == 2, (9, 9)).clone().detach()
            # Swap turn
            state[-1] = 1 - self._vector[-1].clone().detach()
            states.append((state, self.export_save()))
            # print("Torch board:", torch.sum(self.torch_board == 1), "vs", torch.sum(self.torch_board == 2))
            # for i in range(self.tensor_size):
            #     print("i=", i, torch.sum(self._vector[i] == 1))
            # input("State exploration")
            self.board.pop()
        return actions, states

    def evaluate(self, state) -> Tuple[np.ndarray, float]:
        """
        Evaluate the specified state, which is a torch tensor.
        """
        state = state.view(1, *state.shape).float()

        p, v = self.model(state.to(self.device))

        p = torch.softmax(p, dim=1).detach().cpu().numpy()[0]

        v = v.detach().cpu().numpy()[0][0]
        return p, v

    def reset(self):
        """
        Reset and init the current game.
        """
        self._vector = torch.zeros((self.tensor_size, 9, 9), dtype=int)
        self._vector[0, :, :] = torch.reshape(self.torch_board == 1, (9, 9)).clone().detach()
        self._vector[1, :, :] = torch.reshape(self.torch_board == 2, (9, 9)).clone().detach()

    def get_winner(self) -> bool:
        # TODO: Replace by Taylor-Tromp score
        (black, white) = self.board.compute_score()
        if black > white:
            return self._mycolor == Goban.Board._BLACK
        else:
            return self._mycolor == Goban.Board._WHITE

    def export_save(self):
        self.board._pushBoard()
        save = self.board._trailMoves.pop()
        save.append(self.board._historyMoveNames.copy())
        save.append(self.board._seenHashes.copy())
        return save

    def load_save(self, save: List):
        self.board._seenHashes = save[-1].copy()
        self.board._historyMoveNames = save[-2].copy()
        self.board._currentHash = save[-3]
        self.board._empties = save[-4].copy()
        self.board._stringSizes = save[-5].copy()
        self.board._stringLiberties = save[-6].copy()
        self.board._stringUnionFind = save[-7].copy()
        self.board._lastPlayerHasPassed = save[-8]
        self.board._gameOver = save[-9]
        self.board._board = save[-10].copy()
        self.board._nextPlayer = save[-11]
        self.board._capturedBLACK = save[-12]
        self.board._capturedWHITE = save[-13]
        self.board._nbBLACK = save[-14]
        self.board._nbWHITE = save[-15]

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")
