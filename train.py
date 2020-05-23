import time
from torch.multiprocessing import Manager, Queue
from torch.utils.tensorboard import SummaryWriter

from config import OthelloConfig
from util import ReplayBuffer, SelfPlayWorker, TrainingWorker


def train(experiment: int, batch: int, resume: bool):
    cfg = OthelloConfig(experiment, batch)
    manager = Manager()
    buffer = manager.list()
    replay_buffer = ReplayBuffer(buffer)
    shared_state_dicts = manager.dict()
    message_queue = Queue()
    log_queue = Queue()  # a single log is dictionary
    writer = SummaryWriter(cfg.dir_log)
    training_worker = TrainingWorker("Training Worker", message_queue, log_queue, shared_state_dicts, replay_buffer,
                                     cfg.device_name_tw, cfg, resume)
    self_play_workers = []
    for i in range(cfg.num_self_play_workers):
        self_play_workers.append(SelfPlayWorker("Self-Play Worker-" + str(i), message_queue, log_queue,
                                                shared_state_dicts, replay_buffer, cfg.device_names_sp[i], cfg))
    print("Starting training...")
    training_worker.start()
    for worker in self_play_workers:
        worker.start()
    print("Training started.")
    try:
        while training_worker.is_alive():
            if log_queue.empty():
                time.sleep(1.0)
                continue
            log = log_queue.get()
            for k, v in log.items():
                if k is "gs":
                    continue
                writer.add_scalar(k, v, log["gs"])
            del log
    except KeyboardInterrupt:
        print("KeyboardInterrupt, stopping training...")
    finally:
        for i in range(cfg.num_self_play_workers + 1):
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
