from __future__ import annotations

from typing import Union, OrderedDict, Dict

import numpy as np
import pickle
import time
import torch
from torch.multiprocessing import Process, Queue

from config import OthelloConfig
from utils.game import Othello
from utils.player import AZPlayer
from utils.model import Network
from utils.util import ReplayBuffer, image_to_tensor, Node, mcts, generate_training_data, calculate_loss
from vmcts.vmcts import VMCTSPlayer


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
        self._network = Network()
        self._game = Othello(self._cfg)
        self._interrupted = False

    def run(self):
        print(super().name, "started.")
        self._network.to(self._device).eval()
        while True:
            self._load_latest_network()
            self._game.reset()
            target_policies = []
            node, *_ = Node.get_new_node(self._cfg, self._game, self._network, self._device)
            while not self._game.is_terminal():
                self._check_message_queue()
                if self._interrupted:
                    break
                for _ in range(self._cfg.num_simulations):
                    mcts(node, self._cfg, self._network, self._device)
                target_policy = node.get_policy()
                action = node.select_optimal_action()
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
            if self._cfg.debug:
                print(super().name, "completed one simulation.")
        print(super().name, "terminated.")

    def _check_message_queue(self):
        if not self._message_queue.empty():
            msg = self._message_queue.get()
            if msg == self._cfg.message_interrupt:
                self._interrupted = True

    # noinspection DuplicatedCode
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
        self._network = Network()
        # noinspection PyUnresolvedReferences
        self._optim = torch.optim.RMSprop(
            self._network.parameters(), lr=self._cfg.learning_rate_schedule[1], weight_decay=self._cfg.weight_decay
        )
        self._gs = 1
        self._interrupted = False

    def run(self):
        print(super().name, "started.")
        self._network.to(self._device).train()
        # noinspection PyUnresolvedReferences
        self._optim = torch.optim.RMSprop(
            self._network.parameters(), lr=self._cfg.learning_rate_schedule[1], weight_decay=self._cfg.weight_decay
        )
        if self._resume:
            self._load_parameters()
        self._flush_network()
        for epoch in range(self._gs, self._cfg.training_steps + 1):
            self._check_message_queue()
            self._check_replay_buffer()
            if self._interrupted:
                break
            self._reschedule_lr()
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
                type: "scalar",
                "losses/policy_loss": policy_loss.item(),
                "losses/value_loss": value_loss.item(),
                "losses/total_loss": total_loss.item(),
                "gs": self._gs
            }
            self._log_queue.put(log)
            self._gs = self._gs + 1
            if epoch % self._cfg.checkpoint_interval == 0:
                self._save_parameters()
            if self._cfg.debug:
                print(log)
        print(super().name, "terminated.")

    def _load_parameters(self):
        print("Loading parameters...")
        with open(self._cfg.dir_gs, "rb") as f:
            self._gs = pickle.load(f)
        self._network.load_state_dict(torch.load(self._cfg.dir_network, map_location=self._device))
        self._optim.load_state_dict(torch.load(self._cfg.dir_optim, map_location=self._device))
        self._network.train()
        print("Parameters loaded successfully.")

    def _save_parameters(self):
        with open(self._cfg.dir_gs, "wb") as f:
            pickle.dump(self._gs, f)
        torch.save(self._network.state_dict(), self._cfg.dir_network)
        torch.save(self._optim.state_dict(), self._cfg.dir_optim)

    def _reschedule_lr(self):
        if self._gs in self._cfg.learning_rate_schedule.keys():
            # noinspection PyUnresolvedReferences
            self._optim = torch.optim.RMSprop(
                self._network.parameters(), lr=self._cfg.learning_rate_schedule[self._gs],
                weight_decay=self._cfg.weight_decay
            )

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


class EvaluationWorker(Process):
    def __init__(
            self, name: str, message_queue: Queue, log_queue: Queue,
            shared_state_dicts: Dict[str, Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor], int]],
            device_name: str, cfg: OthelloConfig, resume: bool
    ):
        super().__init__(name=name)
        self._message_queue = message_queue
        self._log_queue = log_queue
        self._shared_state_dicts = shared_state_dicts
        self._cfg = cfg
        self._device = torch.device(device_name)
        self._network = Network()
        self._gs = 1
        self._interrupted = False
        self._resume = resume

    def run(self):
        print(super().name, "started.")
        self._network.to(self._device)
        if self._resume:
            with open(self._cfg.dir_eval_gs, "rb") as f:
                self._gs = pickle.load(f)
        az_first = True
        while True:
            self._check_message_queue()
            if self._interrupted:
                break
            self._load_latest_network()
            az_player = AZPlayer(self._cfg, self._network, self._device)
            vmcts_player = VMCTSPlayer(self._cfg)
            az_turn = True if az_first else False
            while not az_player.game().is_terminal():
                self._check_message_queue()
                if self._interrupted:
                    break
                if az_turn:
                    action = az_player.choose_action()
                    az_player.play(action)
                    vmcts_player.play(action)
                    az_turn = False
                else:
                    action = vmcts_player.choose_action()
                    vmcts_player.play(action)
                    az_player.play(action)
                    az_turn = True
            if self._interrupted:
                break
            winner = az_player.game().winner()
            if (az_first and winner == 0) or ((not az_first) and winner == 1):
                az_score = 1
            else:
                az_score = -1
            log = {
                "type": "scalar",
                "az_score": az_score,
                "gs": self._gs
            }
            self._log_queue.put(log)
            az_first = False if az_first else True
            self._gs = self._gs + 1
            with open(self._cfg.dir_eval_gs, "wb") as f:
                pickle.dump(self._gs, f)
            if self._cfg.debug:
                print(log)
        print(super().name, "terminated.")

    def _check_message_queue(self):
        if not self._message_queue.empty():
            msg = self._message_queue.get()
            if msg == self._cfg.message_interrupt:
                self._interrupted = True

    # noinspection DuplicatedCode
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
