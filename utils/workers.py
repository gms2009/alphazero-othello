from __future__ import annotations

from typing import Union, OrderedDict, Dict

import numpy as np
import pickle
import time
import torch
from torch.multiprocessing import Process, Queue

from config import OthelloConfig
from utils.game import Othello
from utils.model import Network
from utils.util import (ReplayBuffer, image_to_tensor, filter_legal_action_probs, Node, mcts,
                        generate_training_data, calculate_loss)


class SelfPlayWorker(Process):
    def __init__(
            self, name: str, message_queue: Queue, log_queue: Queue,
            shared_state_dicts: Dict[str, Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor], int]],
            replay_buffer: ReplayBuffer, device_name: str, cfg: OthelloConfig
    ):
        super().__init__(name=name)
        self._message_queue = message_queue
        self._log_queue = log_queue
        self._shared_state_dicts = shared_state_dicts
        self._replay_buffer = replay_buffer
        self._cfg = cfg
        self._device = torch.device(device_name)
        self._network = Network().to(self._device).eval()
        self._game = Othello()
        self._interrupted = False

    def run(self):
        print(super().name, "started.")
        while True:
            self._load_latest_network()
            self._game.reset()
            target_policies = []
            state_tensor = image_to_tensor(self._game.make_input_image(self._cfg), self._device)
            with torch.no_grad():
                p, v = self._network.inference(state_tensor)
                actions_mask = torch.as_tensor(self._game.legal_actions_mask(), dtype=torch.float32).to(self._device)
                p = filter_legal_action_probs(p.unsqueeze(0), actions_mask.unsqueeze(0))
                p.squeeze(0)
            p, v = p.cpu().numpy(), v.cpu().numpy()
            node = Node(self._cfg, self._game, p)
            while not self._game.is_terminal():
                self._check_message_queue()
                if self._interrupted:
                    break
                for _ in range(self._cfg.num_simulations):
                    mcts(node, self._cfg, self._network, self._device)
                target_policy = node.get_policy()
                action = int(target_policy.argmax())
                target_policies.append(target_policy)
                child = node.child(action)
                self._game = child.game()
                node = child
            if self._interrupted:
                break
            final_returns = np.array(self._game.returns()).astype(np.float32)
            target_policies = np.array(target_policies).astype(np.float32)
            training_data = generate_training_data(self._cfg, self._game, target_policies, final_returns)
            self._replay_buffer.save_training_data(training_data)
        print(super().name, "terminated.")

    def _check_message_queue(self):
        if not self._message_queue.empty():
            msg = self._message_queue.get()
            if msg == self._cfg.message_interrupt:
                self._interrupted = True

    def _load_latest_network(self):
        while True:
            try:
                state_dict = self._shared_state_dicts["network"]
                for k, v in state_dict.items():
                    state_dict[k] = v.to(self._device)
                self._network.load_state_dict(state_dict)
                self._network.eval()
                return
            except KeyError:
                pass
            self._check_message_queue()
            if self._interrupted:
                return
            time.sleep(1.0)


class TrainingWorker(Process):
    def __init__(
            self, name: str, message_queue: Queue, log_queue: Queue,
            shared_state_dicts: Dict[str, Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor], int]],
            replay_buffer: ReplayBuffer, device_name: str, cfg: OthelloConfig, resume: bool
    ):
        super().__init__(name=name)
        self._message_queue = message_queue
        self._log_queue = log_queue
        self._shared_state_dicts = shared_state_dicts
        self._replay_buffer = replay_buffer
        self._cfg = cfg
        self._resume = resume
        self._device = torch.device(device_name)
        self._network = Network().to(self._device).train()
        # noinspection PyUnresolvedReferences
        self._optim = torch.optim.RMSprop(
            self._network.parameters(), lr=self._cfg.learning_rate_schedule[30000], weight_decay=self._cfg.weight_decay
        )
        self._gs = 1
        self._interrupted = False

    def run(self):
        print(super().name, "started.")
        if self._resume:
            self._load_parameters()
        self._flush_network()
        for epoch in range(self._gs, self._cfg.training_steps + 1):
            self._check_message_queue()
            self._check_replay_buffer()
            if self._interrupted:
                break
            images, target_action_probs, target_values, action_masks = self._replay_buffer.sample_batch()
            images = image_to_tensor(images, self._device)
            target_action_probs = torch.as_tensor(target_action_probs, dtype=torch.float32).to(self._device)
            target_values = torch.as_tensor(target_values, dtype=torch.float32).to(self._device)
            self._optim.zero_grad()
            predicted_action_probs, predicted_values = self._network(images)
            policy_loss, value_loss, total_loss = calculate_loss(predicted_action_probs, predicted_values,
                                                                 target_action_probs, target_values)
            total_loss.backward()
            self._optim.step()
            self._flush_network()
            log = {
                "losses/policy_loss": policy_loss.item(),
                "losses/value_loss": value_loss.item(),
                "losses/total_loss": total_loss.item(),
                "gs": self._gs
            }
            self._log_queue.put(log)
            self._gs = self._gs + 1
            if epoch % self._cfg.checkpoint_interval == 0:
                self._save_parameters()
        print(super().name, "terminated.")

    def _load_parameters(self):
        print("Loading parameters...")
        with open(self._cfg.dir_gs, "rb") as f:
            self._gs = pickle.load(f)
        self._network.load_state_dict(torch.load(self._cfg.dir_network, map_location=self._device))
        self._optim.load_state_dict(torch.load(self._cfg.dir_optim, map_location=self._device))
        print("Parameters loaded successfully.")

    def _save_parameters(self):
        with open(self._cfg.dir_gs, "wb") as f:
            pickle.dump(self._gs, f)
        torch.save(self._network.state_dict(), self._cfg.dir_network)
        torch.save(self._optim.state_dict(), self._cfg.dir_optim)

    def _flush_network(self):
        network_state_dict = self._network.state_dict()
        for k, v in network_state_dict.items():
            network_state_dict[k] = v.detach().cpu()
        self._shared_state_dicts["network"] = network_state_dict

    def _check_message_queue(self):
        if not self._message_queue.empty():
            msg = self._message_queue.get()
            if msg == self._cfg.message_interrupt:
                self._interrupted = True

    def _check_replay_buffer(self):
        while self._replay_buffer.empty():
            self._check_message_queue()
            if self._interrupted:
                return
            time.sleep(1.0)
