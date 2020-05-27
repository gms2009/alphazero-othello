import numpy as np

from config import OthelloConfig
from utils.game import Othello
from vmcts.vmcts import VMCTSNode


class VMCTSPlayer(object):
    def __init__(self, cfg: OthelloConfig):
        self._name = "Vanilla MCTS"
        self._cfg = cfg
        self._game = Othello(self._cfg)
        self._node = VMCTSNode(self._cfg, self._game)

    def name(self) -> str:
        return self._name

    def game(self) -> Othello:
        return self._game

    def play(self, action: int):
        if action not in self._game.legal_actions():
            raise ValueError(str(action) + " is invalid move")
        child_node = self._node.child(action)
        if child_node is None:
            child_game = self._game.clone()
            child_game.apply_action(action)
            child_node = VMCTSNode(self._cfg, child_game)
        self._node = child_node
        self._game = self._node.game()

    def choose_action(self) -> int:
        if self._game.is_terminal():
            return -1
        for sim in range(self._cfg.num_simulations_vmcts):
            vmcts(self._node, self._cfg)
        action = self._node.select_optimal_action()
        return action


def vmcts(node: VMCTSNode, cfg: OthelloConfig) -> np.ndarray:
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
    child = VMCTSNode(cfg, child_game)
    node.update_child(child, action)
    returns = vmcts(child, cfg)
    node.add_returns(action, returns)
    return returns
