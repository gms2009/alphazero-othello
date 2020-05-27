from __future__ import annotations

import torch

from config import OthelloConfig
from players.azplayer import AZPlayer
from utils.model import Network
from players.humanplayer import HumanPlayer
from players.vmctsplayer import VMCTSPlayer

player_choices = ["human", "vmcts", "self"]


def evaluate(vs: str, games: int, experiment: int, batch: int):
    score_p1 = 0
    score_p2 = 0
    cfg = OthelloConfig(experiment, batch)
    device = torch.device(cfg.device_name_eval_player)
    network = Network().to(device).eval()
    network.load_state_dict(torch.load(cfg.dir_network, map_location=device))
    az_first = False
    for g in range(1, games + 1):
        print("Playing game %d of %d." % (g, games))
        p1 = AZPlayer(cfg, network, device)
        if vs == "human":
            p2 = HumanPlayer(cfg)
        elif vs == "vmcts":
            p2 = VMCTSPlayer(cfg)
        else:
            p2 = AZPlayer(cfg, network, device)
        az_turn = az_first
        while not p1.game().is_terminal():
            print(p1.game())
            print("Available actions:", list(p1.game().legal_actions()))
            if az_turn:
                print(p1.name(), " turn. Choosing action...")
                action = p1.choose_action()
                az_turn = False
            else:
                print(p1.name(), " turn. Choosing action...")
                action = p2.choose_action()
                az_turn = True
            print("Chosen action:", action)
            p1.play(action)
            p2.play(action)
        print(p1.game())
        winner = p1.game().winner()
        if winner == 2:
            print("Game draw.")
        if (az_first and winner == 0) or (not az_first and winner == 1):
            score_p1 += 1
            print("\tGame winner:", p1.name())
        else:
            score_p2 += 1
            print("\tGame winner:", p2.name())
        print("Total scores-")
        print("\t", p1.name(), ":", score_p1, ", ", p2.name(), ":", score_p2)
        az_first = True if not az_first else False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--vs", type=str, choices=player_choices, required=True)
    parser.add_argument("--experiment", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()

    evaluate(args.vs, args.games, args.experiment, args.batch)
