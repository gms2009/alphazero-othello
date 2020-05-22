import numpy as np
import os
import torch
from torch.multiprocessing import Manager, Queue

from config import OthelloConfig
from game import Othello
from model import Network
from util import ReplayBuffer, SelfPlayWorker, TrainingWorker


def train(experiment: int, batch: int, resume: bool):
    cfg = OthelloConfig()
    manager = Manager()
    buffer = manager.list()
    replay_buffer = ReplayBuffer(buffer)
    shared_state_dicts = manager.dict()
    message_queue = Queue()
    device_names_sp = ["cpu"] * cfg.num_self_play_workers
    device_name_tw = "cuda"
    training_worker = TrainingWorker(message_queue, shared_state_dicts, replay_buffer, device_name_tw, experiment,
                                     batch, resume)
    self_play_workers = []
    for i in range(cfg.num_self_play_workers):
        self_play_workers.append(SelfPlayWorker(message_queue, shared_state_dicts, replay_buffer, device_names_sp[i]))
    try:
        training_worker.start()
        for worker in self_play_workers:
            worker.start()
        training_worker.join()
    finally:
        for i in range(cfg.num_self_play_workers+1):
            message_queue.put(cfg.message_interrupt)
        training_worker.join()
        for worker in self_play_workers:
            worker.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--resume", type=bool, default=False)
    args = parser.parse_args()
    train(args.experiment, args.batch, args.resume)
