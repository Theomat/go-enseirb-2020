# -*- coding: utf-8 -*-
from mcts import Node
import time
import Goban
from random import choice
from playerInterface import PlayerInterface
from model import AlphaGoCnn

import torch

import numpy as np


class myPlayer(PlayerInterface):
    '''
    UTC MCTS
    '''

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self.tree = None

        device = "cpu" #"cuda:0" if torch.cuda.is_available() else "cpu"

        print(f"Inference on {'cpu' if device == 'cpu' else torch.cuda.get_device_name(0)}")

        self.net = AlphaGoCnn().to(device)
        self.net.load_state_dict(torch.load("./model.pt", map_location=torch.device(device)))

    def getPlayerName(self):
        return "AlphaGo MCTS"

    def is_winner(self):
        return int(self._board.final_go_score()[0].lower() == Goban.Board.player_name(self._mycolor)[0])

    def predict_all(self, actions, turn):
        self.container = np.zeros((len(actions), 3, 9, 9), dtype=np.float32)
        self.container[:, 2, :, :] = turn
        BLACK = self._mycolor
        WHITE = self._board.flip(self._mycolor)
        to_change = []
        allowed = []
        for i, action in enumerate(actions):
            correct = self._board.push(action)
            allowed.append(correct)
            if not self._board.is_game_over and correct:
                nb = self._board._board
                self.container[i, 0, :, :] = np.reshape(nb == BLACK, [9, 9])
                self.container[i, 1, :, :] = np.reshape(nb == WHITE, [9, 9])
            elif correct:
                to_change.append([i, self.is_winner()])
            self._board.pop()
        probs = self.net(torch.from_numpy(self.container)).detach().numpy()
        for (i, v) in to_change:
            probs[i] = v
        return probs, allowed

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"
        print()
        if self.tree is None:
            self.tree = Node(None)
        turn = 0
        rollouts = 0
        start = time.perf_counter()
        while time.perf_counter() - start <= 5:
            leaf, actions = self.tree.select()
            for action in actions:
                weak_legal = self._board.weak_legal_moves()
                if action not in weak_legal:
                    raise Exception("action became invalid")
                self._board.push(action)
                turn = 1 - turn
            if not self._board.is_game_over():
                legal = self._board.weak_legal_moves()
                priors, allowed = self.predict_all(legal, turn)
                legal = [a for a, t in zip(legal, allowed) if t]
                priors = [a for a, t in zip(priors, allowed) if t]
                leaf.expand(legal, priors)
                value = int(self.rollout())
                rollouts += 1
                leaf.update(value)
            else:
                value = self.is_winner()
                leaf.update(value, closed=True)
            for action in actions:
                self._board.pop()
        node, move, value, incertitude = self.tree.select_move(self._board.legal_moves())
        # New here: allows to consider internal representations of moves
        print("Finished", rollouts, "rollouts !")
        print("I am playing ", self._board.move_to_str(move), "with score:", value, "~", incertitude)
        print("My current board :")
        self._board.prettyPrint()

        self._board.push(move)
        # move is an internal representation. To communicate with the interface I need to change if to a string
        return Goban.Board.flat_to_name(move)

    def playOpponentMove(self, move):
        print("Opponent played ", move, "i.e. ", move)  # New here
        # the board needs an internal represetation to push the move.  Not a string
        flat = Goban.Board.name_to_flat(move)
        self._board.push(flat)
        if self.tree:
            self.tree = self.tree.move_to(flat)

    def rollout(self):
        n = 0
        while not self._board.is_game_over():
            moves = self._board.legal_moves()
            move = choice(moves)
            self._board.push(move)
            n += 1
        result = self.is_winner()
        for i in range(n):
            self._board.pop()
        return result

    def newGame(self, color):
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)
        self.tree = None

    def endGame(self, winner):
        if self._mycolor == winner:
            print("I won!!!")
        else:
            print("I lost :(!!")
