import random
import numpy as np
import torch
from collections import deque
from multiprocessing import Process, Queue

from config import OthelloConfig
from game import Othello
from model import Network

from typing import List, Tuple


class ReplayBuffer(object):
    def __init__(self, cfg: OthelloConfig, buffer: list):
        self.window_size = cfg.window_size
        self.batch_size = cfg.batch_size
        self.buffer = buffer

    def save_training_data(self, training_data: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]):
        for data_point in training_data:
            if len(self.buffer) >= self.window_size:
                self.buffer.pop(0)
            self.buffer.append(data_point)

    def sample_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        samples = random.sample(self.buffer, k=self.batch_size)
        image, pi, z, actions_mask = zip(*samples)
        return (np.array(image).astype(np.bool), np.array(pi).astype(np.float32), 
        np.array(z).astype(np.float32), np.array(actions_mask).astype(np.bool))


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

    def select_action(self) -> int:
        action_mask = self._game.legal_actions_mask()
        ucb = self._c_puct * self._P * (np.sqrt(self._N.sum())/(1+self._N))
        puct = self._Q + ucb
        puct = action_mask * (1+puct) # add 1 for case when all values of puct are zero
        return int(puct.argmax())

    def get_policy(self) -> np.ndarray:
        tau = self._tau_initial if len(self._game) <= self._num_sampling_moves else self._tau_final
        action_mask = self._game.legal_actions_mask()
        N = self._N**tau
        if N.sum() == 0:
            N += 1e-8
        policy = action_mask * (N/N.sum())
        return policy

    def add_returns(self, action: int, returns: np.ndarray):
        self._W[action] += returns[self._game.current_player]
        self._N[action] += 1
        self._Q[action] = self._W[action]/self._N[action]


class SelfPlayWorker(Process):
    def __init__(self, message_queue: Queue, network_list: list, replay_buffer: ReplayBuffer, device_name: str):
        self._message_queue = message_queue
        self._replay_buffer = replay_buffer
        self._device = torch.device(device_name)
        self._network = Network().to(self._device).eval()
        self._game = Othello()
        self._cfg = OthelloConfig()

    def run(self):
        interrupted = False
        while True:
            if not self._message_queue.empty():
                msg = self._message_queue.get()
                if msg == self._cfg.message_interrupt:
                    interrupted = True
                del msg
            if interrupted:
                break
            self._game.reset()
            target_policies = []
            state_tensor = image_to_tensor(self._game.make_input_image(), self._device)
            with torch.no_grad():
                p, v = self._network.inference(state_tensor)
            p, v = p.cpu().numpy(), v.cpu().numpy()
            node = Node(self._cfg, self._game, p)
            while not self._game.is_terminal():
                for _ in range(self._cfg.num_simulations):
                    mcts(node, self._cfg, self._network, self._device)
                target_policy = node.get_policy()
                action = int(target_policy.argmax())
                child = node.child(action)
                target_policies.append(target_policy)
                self._game = child.game()
                node = child
            final_returns = np.array(self._game.returns()).astype(np.float32)
            target_policies = np.array(target_policies).astype(np.float32)
            training_data = generate_training_data(self._cfg, self._game, target_policies, final_returns)
            self._replay_buffer.save_training_data(training_data)
        print(super().pid, "terminated.")


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(image, dtype=torch.float32).to(device)
    return tensor


def generate_training_data(
    cfg: OthelloConfig, game, target_policies: np.ndarray, final_returns: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    assert len(target_policies) == len(game)
    dq = deque(maxlen=cfg.total_input_channels//2)
    training_data = []  # list of (input_image, pi, z, action_mask)
    for _ in range(cfg.total_input_channels//2):
        dq.appendleft(np.zeros((2, 8, 8), dtype=np.bool))
    for i in range(len(game)):
        img = game.history_state(i)
        player = game.history_player(i)
        action_mask = game.history_actions_mask(i)
        dq.appendleft(img)
        x = np.zeros((cfg.total_input_channels, 8, 8), dtype=np.bool)
        for ch, img in enumerate(dq):
            x[ch] += img[0]
            x[(cfg.total_input_channels//2)+ch] += img[1]
        x[-1] += bool(player)
        training_data.append((x, target_policies[i], float(final_returns[player]), action_mask))
    return training_data


def mcts(node: Node, cfg: OthelloConfig, network: Network, device: torch.device) -> np.ndarray:
    if Node.game().is_terminal():
        return Node.game().returns()
    action = Node.select_action()
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
