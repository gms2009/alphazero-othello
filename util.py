import random
import numpy as np
from collections import deque

from config import OthelloConfig
from game import Othello

from typing import List, Tuple


def generate_training_data(cfg: OthelloConfig, g: Othello, target_pis: np.array, final_returns: np.array) -> List[Tuple[np.array, np.array, float]]:
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


class ReplayBuffer(object):
    def __init__(self, cfg: OthelloConfig, buffer: list):
        self.window_size = cfg.window_size
        self.batch_size = cfg.batch_size
        self.buffer = buffer

    def save_training_data(self, training_data: List[Tuple[np.array, np.array, float]]):
        for data_point in training_data:
            if len(self.buffer) >= self.window_size:
                self.buffer.pop(0)
            self.buffer.append(data_point)

    def sample_batch(self) -> Tuple[np.array, np.array, np.array]:
        samples = random.sample(self.buffer, k=self.batch_size)
        image, pi, z = zip(*samples)
        return image, pi, z
