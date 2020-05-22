from __future__ import annotations
from typing import List, Tuple, Union, OrderedDict, Dict

from collections import deque
import numpy as np
import pickle
import random
import time
import torch
from torch.multiprocessing import Process, Queue
from torch.utils.tensorboard import SummaryWriter

from config import OthelloConfig
from game import Othello
from model import Network


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
        samples = random.sample(self._buffer, k=self._batch_size)
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

    def child(self, action: int) -> Union[Node, None]:
        return self._children[action]

    def update_child(self, child: Union[Node, None], action: int):
        self._children[action] = child

    def game(self) -> Othello:
        return self._game

    def select_action(self) -> int:
        action_mask = self._game.legal_actions_mask()
        ucb = self._c_puct * self._P * (np.sqrt(self._N.sum()) / (1 + self._N))
        puct = self._Q + ucb
        puct = action_mask * (1 + puct)  # add 1 for case when all values of puct are zero
        return int(puct.argmax())

    def get_policy(self) -> np.ndarray:
        tau = self._tau_initial if len(self._game) <= self._num_sampling_moves else self._tau_final
        n = self._N ** tau
        if n.sum() == 0:
            n += 1e-8
        policy = n / n.sum()
        return policy

    def add_returns(self, action: int, returns: np.ndarray):
        self._W[action] += returns[self._game.current_player]
        self._N[action] += 1
        self._Q[action] = self._W[action] / self._N[action]


class SelfPlayWorker(Process):
    def __init__(
            self, message_queue: Queue,
            shared_state_dicts: Dict[str, Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor], int]],
            replay_buffer: ReplayBuffer, device_name: str
    ):
        super().__init__()
        self._message_queue = message_queue
        self._shared_state_dicts = shared_state_dicts
        self._replay_buffer = replay_buffer
        self._cfg = OthelloConfig()
        self._device = torch.device(device_name)
        self._network = Network().to(self._device).eval()
        self._game = Othello()

    def run(self):
        interrupted = False
        while True:
            try:
                state_dict = self._shared_state_dicts["network"]
                for k, v in state_dict.items():
                    state_dict[k] = v.to(self._device)
                self._network.load_state_dict(state_dict)
            finally:
                del state_dict
            self._game.reset()
            target_policies = []
            state_tensor = image_to_tensor(self._game.make_input_image(self._cfg), self._device)
            with torch.no_grad():
                p, v = self._network.inference(state_tensor)
            p, v = p.cpu().numpy(), v.cpu().numpy()
            node = Node(self._cfg, self._game, p)
            while not self._game.is_terminal():
                if not self._message_queue.empty():
                    msg = self._message_queue.get()
                    if msg == self._cfg.message_interrupt:
                        interrupted = True
                    del msg
                if interrupted:
                    break
                for _ in range(self._cfg.num_simulations):
                    mcts(node, self._cfg, self._network, self._device)
                target_policy = node.get_policy()
                action = int(target_policy.argmax())
                child = node.child(action)
                target_policies.append(target_policy)
                self._game = child.game()
                node = child
            if interrupted:
                break
            final_returns = np.array(self._game.returns()).astype(np.float32)
            target_policies = np.array(target_policies).astype(np.float32)
            training_data = generate_training_data(self._cfg, self._game, target_policies, final_returns)
            self._replay_buffer.save_training_data(training_data)
        print("SelfPlayWorker-", super().name, "terminated.")


class TrainingWorker(Process):
    def __init__(
            self, message_queue: Queue,
            shared_state_dicts: Dict[str, Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor], int]],
            replay_buffer: ReplayBuffer, device_name: str, experiment: int, batch: int, resume: bool
    ):
        super().__init__()
        self._message_queue = message_queue
        self._shared_state_dicts = shared_state_dicts
        self._replay_buffer = replay_buffer
        self._cfg = OthelloConfig(experiment, batch)
        self._device = torch.device(device_name)
        self._network = Network().to(self._device).train()
        # noinspection PyUnresolvedReferences
        self._optim = torch.optim.RMSprop(
            self._network.parameters(), lr=self._cfg.learning_rate_schedule[0], weight_decay=self._cfg.weight_decay
        )
        self._gs = 1
        if resume:
            print("Loading parameters to resume training...")
            with open(self._cfg.dir_gs, "rb") as f:
                self._gs = pickle.load(f)
            self._network.load_state_dict(torch.load(self._cfg.dir_network, map_location=self._device))
            self._optim.load_state_dict(torch.load(self._cfg.dir_optim, map_location=self._device))
        self._writer = SummaryWriter(self._cfg.dir_log)
        print("Training worker created.\nWriting state dicts to shared_state_dicts...")
        self._shared_state_dicts["network"] = self.network_state_dict()
        print("Successfully written state dicts.")

    def run(self):
        for epoch in range(self._cfg.training_steps):
            interrupted = False
            if not self._message_queue.empty():
                msg = self._message_queue.get()
                if msg == self._cfg.message_interrupt:
                    interrupted = True
                del msg
            if interrupted:
                break
            while self._replay_buffer.empty():
                time.sleep(1.0)
            images, target_action_probs, target_values, action_masks = self._replay_buffer.sample_batch()
            images = image_to_tensor(images, self._device)
            target_action_probs = torch.as_tensor(target_action_probs, dtype=torch.float32).to(self._device)
            target_values = torch.as_tensor(target_values, dtype=torch.float32).to(self._device)
            action_masks = torch.as_tensor(action_masks, dtype=torch.float32).to(self._device)
            self._optim.zero_grad()
            predicted_action_probs, predicted_values = self._network(images)
            predicted_values = filter_legal_action_probs(predicted_action_probs, action_masks)
            policy_loss, value_loss, total_loss = calculate_loss(predicted_action_probs, predicted_values,
                                                                 target_action_probs, target_values)
            total_loss.backward()
            self._optim.step()
            self._shared_state_dicts["network"] = self.network_state_dict()
            self._writer.add_scalar("losses/policy_loss", policy_loss.item(), self._gs)
            self._writer.add_scalar("losses/value_loss", value_loss.item(), self._gs)
            self._writer.add_scalar("losses/total_loss", total_loss.item(), self._gs)
            self._gs += 1
            if (epoch + 1) % self._cfg.checkpoint_interval == 0:
                with open(self._cfg.dir_gs, "wb") as f:
                    pickle.dump(self._gs, f)
                torch.save(self._network.state_dict(), self._cfg.dir_network)
                torch.save(self._optim.state_dict(), self._cfg.dir_optim)
        print("TrainingWorker terminated.")

    def network_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        network_state_dict = self._network.state_dict()
        for k, v in network_state_dict.items():
            network_state_dict[k] = v.cpu()
        return network_state_dict


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
    policy_loss = target_action_probs.T @ torch.log(predicted_action_probs)
    final_loss = value_loss - policy_loss
    return -policy_loss.mean(), value_loss.mean(), final_loss.mean()


def generate_training_data(
        cfg: OthelloConfig, game, target_policies: np.ndarray, final_returns: np.ndarray
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
    game = node.game().clone()
    game.apply_action(action)
    state_tensor = image_to_tensor(game.make_input_image(cfg), device)
    with torch.no_grad():
        p, v = network.inference(state_tensor)
    p, v = p.cpu().numpy(), v.cpu().numpy()
    child = Node(cfg, game, p)
    node.update_child(child, action)
    returns = np.empty(2, dtype=np.float32)
    returns[game.current_player()] = v
    returns[1 - game.current_player()] = -v
    node.add_returns(action, returns)
    return returns
