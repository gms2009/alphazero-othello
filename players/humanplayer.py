from config import OthelloConfig
from utils.game import Othello


class HumanPlayer(object):
    def __init__(self, cfg: OthelloConfig):
        self._name = "Human"
        self._cfg = cfg
        self._game = Othello(self._cfg)

    def name(self) -> str:
        return self._name

    def game(self) -> Othello:
        return self._game

    def play(self, action: int):
        if action not in self._game.legal_actions():
            raise ValueError(str(action) + " is invalid move")
        self._game.apply_action(action)

    def choose_action(self) -> int:
        if self._game.is_terminal():
            return -1
        action = int(input("Choose an action: "))
        return action
