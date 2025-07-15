# BASE_RUN_NAME = f"test_sweep"
# BASE_RUN_NAME = f"train_lvae_100b_debug"
# BASE_RUN_NAME = f"train_lvae_100b"
# BASE_RUN_NAME = f"train_lvae_dist_debug"
# BASE_RUN_NAME = f"train_lvae_dist_debug"
# BASE_RUN_NAME = f"train_lvae_dist_sweep"
# BASE_RUN_NAME = f"train_lvae_dist_cands"

# ["main_lvae.py", "", 1, 1, 48, 16, 512, 1e-4],
# ["main_lvae.py", "", 1, 1, 64, 12, 512, 1e-4],
# ["main_lvae.py", "", 1, 1, 96, 8, 512, 1e-4],
# ["main_lvae.py", "", 1, 1, 128, 6, 512, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 1, 4, 64, 1, 512, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 2, 4, 64, 1, 512, 1e-4],
# ["run_distributed_training_no_hydra.py", "", 1, 4, 64, 1, 512, 1e-4],
# ["run_distributed_training_no_hydra.py", "", 2, 4, 64, 1, 512, 1e-4],

# post idk what happened with the dist debugging ...
# ["main_lvae.py", "", 1, 1, 96, 8, 512, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 1, 1, 96, 8, 512, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 1, 4, 96, 2, 512, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 2, 4, 96, 1, 512, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 1, 4, 96, 8, 512, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 2, 4, 96, 8, 512, 1e-4],
# nothing has run beyond 2N at the moment
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl compile_model=False", 4, 4, 96, 8, 512, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl compile_model=False", 8, 4, 96, 8, 512, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl compile_model=True", 4, 4, 64, 12, 512, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl compile_model=True", 8, 4, 64, 12, 512, 1e-4],