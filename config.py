class OthelloConfig(object):
    def __init__(self):
        # Self-Play
        self.num_actors = 12
        self.num_sampling_moves = 10
        self.max_moves = 512
        self.num_simulations = 500

        # Root prior expoloration noise
        self.root_dirichlet_alpha = 0.5
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.ucb_c_puct = 4

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
