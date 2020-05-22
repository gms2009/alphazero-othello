import os


class OthelloConfig(object):
    def __init__(self, experiment: int = 1, batch: int = 1):
        # Self-Play
        self.num_self_play_workers = 12
        self.max_moves = 512
        self.num_simulations = 500

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
        self.training_steps = int(100e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e5)
        self.batch_size = 512

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Schedule
        self.learning_rate_schedule = {
            0: 2e-1,
            10e3: 2e-2,
            30e3: 2e-3,
            50e3: 2e-4
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
