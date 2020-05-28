from __future__ import annotations
from typing import List, Tuple, Union

from collections import deque
import numpy as np
import torch

from config import OthelloConfig
from utils.game import Othello
from utils.model import Network


class ReplayBuffer(object):
    def __init__(self, buffer: list):
        cfg = OthelloConfig()
        self._window_size = cfg.window_size
        self._batch_size = cfg.batch_size
        self._buffer = buffer

    def __len__(self) -> int:
        return len(self._buffer)

    def empty(self) -> bool:
        return len(self) < self._batch_size

    def save_training_data(self, training_data: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]):
        for data_point in training_data:
            if len(self._buffer) >= self._window_size:
                self._buffer.pop(0)
            self._buffer.append(data_point)

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = [int(i) for i in np.random.choice(range(len(self)), size=self._batch_size, replace=False)]
        samples = []
        for i in indices:
            samples.append(self._buffer[i])
        image, pi, z, actions_mask = zip(*samples)
        return (np.array(image).astype(np.bool), np.array(pi).astype(np.float32),
                np.expand_dims(np.array(z).astype(np.float32), -1), np.array(actions_mask).astype(np.bool))


class Node(object):
    def __init__(self, cfg: OthelloConfig, game: Othello, action_probs: np.ndarray):
        self._cfg = cfg
        self._game = game
        self._P = action_probs.copy()
        self._N = np.zeros(self._game.num_distinct_actions(), dtype=np.int64)
        self._Q = np.zeros(self._game.num_distinct_actions(), dtype=np.float32)
        self._W = np.zeros(self._game.num_distinct_actions(), dtype=np.float32)
        self._children = [None for _ in range(self._game.num_distinct_actions())]

    def child(self, action: int) -> Union[Node, None]:
        return self._children[action]

    def update_child(self, child: Union[Node, None], action: int):
        self._children[action] = child

    def game(self) -> Othello:
        return self._game

    def select_action(self) -> int:
        action_mask = self._game.legal_actions_mask()
        ucb = self._cfg.ucb_c_puct * self._P * (np.sqrt(self._N.sum()) / (1 + self._N))
        puct = self._Q + ucb
        puct = action_mask * (1 + puct)  # add 1 for case when all values of puct are zero
        action = int(puct.argmax())
        return action

    def select_optimal_action(self) -> int:
        target_policy = self.get_policy()
        action = int(target_policy.argmax())
        return action

    def get_policy(self) -> np.ndarray:
        tau = self._cfg.tau_initial if len(self._game) <= self._cfg.num_sampling_moves else self._cfg.tau_final
        n = self._N ** (1/tau)
        if n.sum() == 0:
            n += 1e-8
        policy = n / n.sum()
        return policy

    def add_returns(self, action: int, returns: np.ndarray):
        self._W[action] += returns[self._game.current_player()]
        self._N[action] += 1
        self._Q[action] = self._W[action] / self._N[action]

    @staticmethod
    def get_new_node(
            cfg: OthelloConfig, game: Othello, network: Network, device: torch.device
    ) -> Tuple[Node, np.ndarray, np.ndarray]:
        state_tensor = image_to_tensor(game.make_input_image(), device)
        with torch.no_grad():
            p, v = network.inference(state_tensor)
            actions_mask = torch.as_tensor(game.legal_actions_mask(), dtype=torch.float32).to(device)
            p = filter_legal_action_probs(p.unsqueeze(0), actions_mask.unsqueeze(0))
            p.squeeze(0)
        p, v = p.cpu().numpy(), v.cpu().numpy()
        new_node = Node(cfg, game, p)
        return new_node, p, v


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(image, dtype=torch.float32).to(device)
    return tensor


def filter_legal_action_probs(action_probs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    action_probs = action_probs + 0.01  # for case when all legal actions sum to zero
    masked_probs = action_mask * action_probs
    return masked_probs / masked_probs.sum(dim=1, keepdim=True)


def calculate_loss(
        predicted_action_probs: torch.Tensor, predicted_values: torch.Tensor,
        target_action_probs: torch.Tensor, target_values: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value_loss = torch.square(target_values - predicted_values)
    policy_loss = (target_action_probs * torch.log(predicted_action_probs)).sum(dim=1)
    final_loss = value_loss - policy_loss
    return -policy_loss.mean(), value_loss.mean(), final_loss.mean()


def generate_training_data(
        cfg: OthelloConfig, game: Othello, target_policies: np.ndarray, final_returns: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    assert len(target_policies) == len(game)
    dq = deque(maxlen=cfg.total_input_channels // 2)
    training_data = []  # list of (input_image, pi, z, action_mask)
    for _ in range(cfg.total_input_channels // 2):
        dq.appendleft(np.zeros((2, 8, 8), dtype=np.bool))
    for i in range(len(game)):
        img = game.history_state(i)
        player = game.history_player(i)
        action_mask = game.history_actions_mask(i)
        dq.appendleft(img)
        x = np.zeros((cfg.total_input_channels, 8, 8), dtype=np.bool)
        for ch, img in enumerate(dq):
            x[ch] += img[0]
            x[(cfg.total_input_channels // 2) + ch] += img[1]
        x[-1] += bool(player)
        training_data.append((x, target_policies[i], float(final_returns[player]), action_mask))
    return training_data


def mcts(node: Node, cfg: OthelloConfig, network: Network, device: torch.device) -> np.ndarray:
    if node.game().is_terminal():
        return node.game().returns()
    action = node.select_action()
    child = node.child(action)
    if child is not None:
        returns = mcts(child, cfg, network, device)
        node.add_returns(action, returns)
        return returns
    child_game = node.game().clone()
    child_game.apply_action(action)
    if child_game.is_terminal():  # if new created child node is terminal node
        returns = child_game.returns()
        node.add_returns(action, returns)
        p = np.zeros(child_game.num_distinct_actions(), dtype=np.float32)
        child = Node(cfg, child_game, p)  # create an empty node. It will never be in training data.
        node.update_child(child, action)
        return returns
    child, p, v = Node.get_new_node(cfg, child_game, network, device)
    node.update_child(child, action)
    returns = np.empty(2, dtype=np.float32)
    returns[child_game.current_player()] = v
    returns[1 - child_game.current_player()] = -v
    node.add_returns(action, returns)
    return returns
