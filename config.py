import os


class OthelloConfig(object):
    def __init__(self, experiment: int = 1, batch: int = 1):
        self.debug = True

        self.white_piece = "□"
        self.black_piece = "■"
        # Self-Play
        self.num_self_play_workers = 5
        self.max_moves = 512
        self.num_simulations = 400
        self.device_names_sp = ["cuda"] * self.num_self_play_workers

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.5
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.ucb_c_puct = 4

        # Target policy params
        self.tau_initial = 1
        self.tau_final = 0.01
        self.num_sampling_moves = 10

        # Training
        self.training_steps = 200000
        self.checkpoint_interval = 1000
        self.window_size = 100000
        self.batch_size = 512
        self.device_name_tw = "cuda"

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Schedule
        self.learning_rate_schedule = {
            1: 4e-3,
            10000: 3e-3,
            30000: 2e-3,
            50000: 1e-3
        }

        # Input
        self.num_history_states = 7
        self.total_input_channels = 17
        self.players = {
            0: "black",
            1: "white"
        }

        # Messages
        self.message_interrupt = 1

        # Directories
        experiment_str = "experiment-" + str(experiment) + "-batch-" + str(batch)
        self.dir_log = os.path.join("logs", experiment_str)
        os.makedirs(self.dir_log, exist_ok=True)
        self.dir_saved_models = os.path.join("saved_models", experiment_str)
        os.makedirs(self.dir_saved_models, exist_ok=True)
        self.dir_gs = os.path.join(self.dir_saved_models, "gs.pkl")
        self.dir_network = os.path.join(self.dir_saved_models, "network.pt")
        self.dir_optim = os.path.join(self.dir_saved_models, "optim.pt")
        self.dir_replay_buffer = os.path.join(self.dir_saved_models, "replay_buffer.pkl")
        self.dir_eval_gs = os.path.join(self.dir_saved_models, "eval_gs.pkl")

        # EvaluationWorker
        self.device_name_ew = "cuda"

        # Vanilla MCTS
        self.vmcts_c_uct = 1.414
        self.num_simulations_vmcts = 500

        # Evaluation
        self.num_simulations_eval_player = 500
        self.device_name_eval_player = "cuda"
