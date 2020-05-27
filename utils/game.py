from __future__ import annotations

import numpy as np
import pyspiel

from config import OthelloConfig


class Othello(object):
    def __init__(self, cfg: OthelloConfig):
        self._cfg = cfg
        self._game = pyspiel.load_game("othello")
        self._state = self._game.new_initial_state()
        self._history = []  # list of [state, current_player, action]

    def __str__(self):
        rep = self._state.__str__()
        rep = rep.replace("x", self._cfg.black_piece)
        rep = rep.replace("o", self._cfg.white_piece)
        rep = rep.replace("t" + self._cfg.white_piece, "to")
        return rep

    def __len__(self) -> int:
        return len(self._history)

    def clone(self) -> Othello:
        g = Othello(self._cfg)
        g._state = self._state.clone()
        history = list(self._history)
        for i in range(len(history)):
            history[i][0] = history[i][0].clone()
        g._history = history
        return g

    def reset(self):
        self._state = self._game.new_initial_state()
        self._history = []

    def num_distinct_actions(self) -> int:
        return self._game.num_distinct_actions()

    def legal_actions(self) -> np.ndarray:
        return np.array(self._state.legal_actions()).astype(np.int64)

    def legal_actions_mask(self) -> np.ndarray:
        mask = np.array(self._state.legal_actions_mask()).astype(np.bool)
        return mask

    def action_to_string(self, action: int) -> str:
        return self._state.action_to_string(action)

    def current_player(self) -> int:
        return self._state.current_player()

    def current_state(self) -> np.ndarray:
        return self._make_image(self._state)

    def history_state(self, index: int) -> np.ndarray:
        return self._make_image(self._history[index][0])

    def history_player(self, index: int) -> int:
        return self._history[index][1]

    def history_action(self, index: int) -> int:
        return self._history[index][2]

    def history_actions_mask(self, index: int) -> np.ndarray:
        return np.array(self._history[index][0].legal_actions_mask()).astype(bool)

    def is_terminal(self) -> bool:
        return self._state.is_terminal()

    def apply_action(self, action: int):
        child = self._state.child(action)
        self._history.append([self._state, self.current_player(), action])
        self._state = child

    def returns(self) -> np.ndarray:
        return np.array(self._state.returns()).astype(np.float32)

    def winner(self) -> int:
        if not self.is_terminal():
            return -1
        ret = self.returns()
        if ret[0] == ret[1]:
            return 2
        return int(np.argmax(ret))

    def make_input_image(self) -> np.ndarray:
        image = np.zeros((self._cfg.total_input_channels, 8, 8), dtype=np.bool)
        image[-1] += bool(self.current_player())
        temp = self.current_state()
        image[0] += temp[0]
        image[self._cfg.num_history_states + 0] += temp[1]
        for i in range(1, self._cfg.num_history_states + 1):
            try:
                temp = self.history_state(-i)
                image[i] += temp[0]
                image[self._cfg.num_history_states + i] += temp[1]
            except IndexError:
                break
        return image

    @staticmethod
    def _make_image(state: pyspiel.State) -> np.ndarray:
        obs = np.array(state.observation_tensor()).reshape((3, 8, 8)).astype(np.bool)
        obs = obs[1:]  # obs channels -> 0:black, 1:white
        return obs
