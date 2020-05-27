from __future__ import annotations
from typing import Union

import numpy as np

from config import OthelloConfig
from utils.game import Othello


class VMCTSNode(object):
    def __init__(self, cfg: OthelloConfig, game: Othello):
        self._cfg = cfg
        self._game = game
        self._N = np.zeros(len(self._game.legal_actions()), dtype=np.int64)
        self._Q = np.zeros(len(self._game.legal_actions()), dtype=np.float32)
        self._W = np.zeros(len(self._game.legal_actions()), dtype=np.float32)
        self._children = [None for _ in range(len(self._game.legal_actions()))]

    def child(self, action: int) -> Union[VMCTSNode, None]:
        index = int(np.argwhere(self._game.legal_actions() == action)[0])
        return self._children[index]

    def update_child(self, child: Union[VMCTSNode, None], action: int):
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
