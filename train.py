import numpy as np
import os
import torch

from config import OthelloConfig
from game import Othello
from model import Network
from util import ReplayBuffer, SelfPlayWorker, TrainingWorker


def train(experiment: int, batch: int, resume: bool):
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--resume", type=bool, default=False)
    args = parser.parse_args()
