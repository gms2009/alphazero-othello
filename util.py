import random
import numpy as np
import torch
from collections import deque

from config import OthelloConfig
from game import Othello
from model import Network

from typing import List, Tuple


class ReplayBuffer(object):
    def __init__(self, cfg: OthelloConfig, buffer: list):
        self.window_size = cfg.window_size
        self.batch_size = cfg.batch_size
        self.buffer = buffer

    def save_training_data(self, training_data: List[Tuple[np.ndarray, np.ndarray, float]]):
        for data_point in training_data:
            if len(self.buffer) >= self.window_size:
                self.buffer.pop(0)
            self.buffer.append(data_point)

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        samples = random.sample(self.buffer, k=self.batch_size)
        image, pi, z = zip(*samples)
        return image, pi, z


class Node(object):
    def __init__(self, cfg: OthelloConfig, game: Othello, action_probs: np.ndarray):
        self._c_puct = cfg.ucb_c_puct
        self._tau_initial = cfg.tau_initial
        self._tau_final = cfg.tau_final
        self._num_sampling_moves = cfg.num_sampling_moves
        self._game = game
        self._P = action_probs.copy()
        self._N = np.zeros(self._game.num_distinct_actions(), dtype=np.int64)
        self._Q = np.zeros(self._game.num_distinct_actions(), dtype=np.float32)
        self._W = np.zeros(self._game.num_distinct_actions(), dtype=np.float32)
        self._children = [None for _ in range(self._game.num_distinct_actions())]

    def child(self, action: int) -> Node:
        return self._children[action]

    def update_child(self, child: Node, action: int):
        self._children[action] = child

    def game(self) -> Othello:
        return self._game

    def sample_action(self) -> int:
        action_mask = self._game.legal_actions_mask()
        ucb = self._c_puct * self._P * (np.sqrt(self._N.sum())/(1+self._N))
        puct = self._Q + ucb
        puct = action_mask * (1+puct) # add 1 for case when all values of puct are zero
        return np.argmax(puct)

    def get_policy(self) -> np.ndarray:
        tau = self._tau_initial if len(self._game) <= self._num_sampling_moves else self._tau_final
        action_mask = self._game.legal_actions_mask()
        N = self._N**tau
        if N.sum() == 0:
            N += 1e-8
        policy = action_mask * (N/N.sum())

    def add_returns(self, action: int, returns: np.ndarray):
        self._W[action] += returns[self._game.current_player]
        self._N[action] += 1
        self._Q[action] = self._W[action]/self._N[action]


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(image, dtype=torch.float32).to(device)
    return tensor


def generate_training_data(cfg: OthelloConfig, g: Othello, target_pis: np.ndarray, final_returns: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    assert len(target_pis) == len(g)
    dq = deque(maxlen=cfg.total_input_channels//2)
    training_data = []  # list of (input_image, pi, z)
    for _ in range(cfg.total_input_channels//2):
        dq.appendleft(np.zeros((2, 8, 8), dtype=np.bool))
    for i in range(len(g)):
        img = g.history_state(i)
        player = g.history_player(i)
        dq.appendleft(img)
        x = np.zeros((cfg.total_input_channels, 8, 8), dtype=np.bool)
        for ch, img in enumerate(dq):
            x[ch] += img[0]
            x[(cfg.total_input_channels//2)+ch] += img[1]
        x[-1] += bool(player)
        training_data.append((x, target_pis[i], final_returns[player]))
    return training_data


def mcts(node: Node, cfg: OthelloConfig, network: Network, device: torch.device) -> np.ndarray:
    if Node.game().is_terminal():
        return Node.game().returns()
    action = Node.sample_action()
    child = Node.child(action)
    if child is not None:
        returns = mcts(child, cfg, network, device)
        node.add_returns(action, returns)
        return returns
    game = Node.game().clone()
    game.apply_action(action)
    state_tensor = image_to_tensor(game.current_state(), device)
    with torch.no_grad():
        p, v = network.inference(state_tensor)
    p, v = p.cpu().numpy(), v.cpu().numpy()
    child = Node(cfg, game, p)
    node.update_child(child, action)
    returns = np.empty(2, dtype=np.float32)
    returns[game.current_player] = v
    returns[1-game.current_player] = -v
    node.add_returns(action, returns)
    return returns
