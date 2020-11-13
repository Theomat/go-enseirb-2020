# -*- coding: utf-8 -*-
from mcts import Node
import time
import Goban
from random import choice
from playerInterface import PlayerInterface


class myPlayer(PlayerInterface):
    '''
    UTC MCTS
    '''

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self.tree = None

    def getPlayerName(self):
        return "MCTS"

    def getPlayerMove(self):
        if self._board.is_game_over():
            print("Referee told me to play but the game is over!")
            return "PASS"
        start = time.perf_counter()

        if self.tree is None:
            self.tree = Node(None, None)
        while time.perf_counter() - start <= 5:
            leaf, actions = self.tree.select()
            for action in actions:
                self._board.push(action)
            if not self._board.is_game_over():
                leaf.expand(self._board.legal_moves())
                value = int(self.rollout())
            else:
                value = int(self._board.final_go_score()[0].lower() == Goban.Board.player_name(self._mycolor)[0])
            leaf.update(value)
            for action in actions:
                self._board.pop()

        node, move, value, incertitude = self.tree.select_move(self._board.legal_moves())
        # New here: allows to consider internal representations of moves
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
        result = self._board.final_go_score()[0].lower() == Goban.Board.player_name(self._mycolor)[0]
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
