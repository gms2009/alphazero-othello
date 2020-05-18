import numpy as np
from collections import deque

from config import OthelloConfig
from game import Othello

from typing import List, Tuple


class ReplayBuffer(object):
    pass


def generate_training_data(g: Othello, cfg: OthelloConfig, target_pis: np.array, final_returns: np.array) -> List[Tuple[np.array, np.array, int]]:
    assert len(target_pis) == len(g)
    dq = deque(maxlen=cfg.total_input_channels//2)
    training_data = []  # list of (input_image, pi, z)
    for _ in range(cfg.total_input_channels//2):
        dq.appendleft(np.zeros((2, 8, 8), dtype=np.bool))
    for i in range(len(g)):
        img = g.history_state(i)
        to_play = g.history_to_play(i)
        dq.appendleft(img)
        x = np.zeros((cfg.total_input_channels, 8, 8), dtype=np.bool)
        for ch, img in enumerate(dq):
            x[ch] += img[0]
            x[(cfg.total_input_channels//2)+ch] += img[1]
        x[-1] += bool(to_play)
        training_data.append((x, target_pis[i], final_returns[to_play]))
    return training_data
