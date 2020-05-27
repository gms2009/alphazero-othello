import torch

from config import OthelloConfig
from utils.game import Othello
from utils.model import Network
from utils.util import Node, mcts


class AZPlayer(object):
    def __init__(self, cfg: OthelloConfig, network: Network, device: torch.device):
        self._name = "AlphaZero"
        self._cfg = cfg
        self._network = network
        self._network.eval()
        self._device = device
        self._game = Othello(self._cfg)
        self._node, *_ = Node.get_new_node(self._cfg, self._game, self._network, self._device)

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
            child_node, *_ = Node.get_new_node(self._cfg, child_game, self._network, self._device)
        self._node = child_node
        self._game = self._node.game()

    def choose_action(self) -> int:
        if self._game.is_terminal():
            return -1
        for sim in range(self._cfg.num_simulations_eval_player):
            mcts(self._node, self._cfg, self._network, self._device)
        action = self._node.select_optimal_action()
        return action
