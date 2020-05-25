from __future__ import annotations
from typing import Union

import numpy as np

from config import OthelloConfig
from utils.game import Othello


class Node(object):
    def __init__(self, cfg: OthelloConfig, game: Othello):
        self._cfg = cfg
        self._game = game
        self._N = np.zeros(len(self._game.legal_actions()), dtype=np.int64)
        self._Q = np.zeros(len(self._game.legal_actions()), dtype=np.float32)
        self._W = np.zeros(len(self._game.legal_actions()), dtype=np.float32)
        self._children = [None for _ in range(len(self._game.legal_actions()))]

    def child(self, action: int) -> Union[Node, None]:
        index = int(np.argwhere(self._game.legal_actions() == action)[0])
        return self._children[index]

    def update_child(self, child: Union[Node, None], action: int):
        index = int(np.argwhere(self._game.legal_actions() == action)[0])
        self._children[index] = child

    def game(self) -> Othello:
        return self._game

    def select_action(self) -> int:
        ucb = self._cfg.vmcts_c_uct * (np.sqrt(self._N.sum()) / (1 + self._N))
        puct = self._Q + ucb
        index = int(puct.argmax())
        return int(self._game.legal_actions()[index])

    def select_optimal_action(self) -> int:
        index = np.argmax(self._N)
        return int(self._game.legal_actions()[index])

    def add_returns(self, action: int, returns: np.ndarray):
        index = int(np.argwhere(self._game.legal_actions() == action)[0])
        self._W[index] += returns[self._game.current_player()]
        self._N[index] += 1
        self._Q[index] = self._W[index] / self._N[index]


class VMCTSPlayer(object):
    def __init__(self, cfg: OthelloConfig):
        self._cfg = cfg
        self._game = Othello()
        self._node = Node(self._cfg, self._game)

    def is_game_terminated(self) -> bool:
        return self._game.is_terminal()

    def winner(self) -> int:
        if not self.is_game_terminated():
            return -1
        returns = self._game.returns()
        return int(np.argmax(returns))

    def print_game(self):
        current_state = self._game.current_state()
        black, white = 0, 1
        legal_actions = self.legal_actions()
        for r in range(8):
            for c in range(8):
                f = self._cfg.empty_piece
                if current_state[black][r][c]:
                    f = self._cfg.black_piece
                if current_state[white][r][c]:
                    f = self._cfg.white_piece
                if (8*r + c) in legal_actions:
                    f = self._cfg.move_piece
                print(f, end=" ")
            print()

    def legal_actions(self):
        return self._game.legal_actions()

    def play(self, action: int):
        if action not in self._game.legal_actions():
            raise ValueError(str(action) + " is invalid move")
        child_node = self._node.child(action)
        if child_node is None:
            child_game = self._game.clone()
            child_game.apply_action(action)
            child_node = Node(self._cfg, child_game)
        self._node = child_node
        self._game = self._node.game()

    def choose_move(self) -> int:
        if self._game.is_terminal():
            return -1
        for sim in range(self._cfg.vmcts_num_simulations):
            vmcts(self._node, self._cfg)
        action = self._node.select_optimal_action()
        return action


def vmcts(node: Node, cfg: OthelloConfig) -> np.ndarray:
    if node.game().is_terminal():
        return node.game().returns()
    action = node.select_action()
    child = node.child(action)
    if child is not None:
        returns = vmcts(child, cfg)
        node.add_returns(action, returns)
        return returns
    child_game = node.game().clone()
    child_game.apply_action(action)
    child = Node(cfg, child_game)
    node.update_child(child, action)
    returns = vmcts(child, cfg)
    node.add_returns(action, returns)
    return returns
