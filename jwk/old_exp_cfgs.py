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

# BASE_RUN_NAME = f"train_lvae_dist_scaling"
# BASE_RUN_NAME = f"train_lvae_dist_prod"
# First try to compile on a single gpu then step it up
# While the mbsz 256 cfg runs at 1N only the mbsz 128 version ran reliably past that
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 1, 1, 256, 8, 128, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 1, 4, 256, 8, 128, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 2, 4, 256, 4, 128, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 4, 4, 256, 2, 128, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 8, 4, 256, 1, 128, 1e-4],
# Trying to get a slightly less intensive setting so that 4N and beyond works
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 1, 1, 128, 2, 128, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 1, 4, 128, 2, 128, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 2, 4, 128, 2, 128, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 4, 4, 128, 2, 128, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 8, 4, 128, 2, 128, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 16, 4, 128, 1, 128, 1e-4],
# best so far in time to soln
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 4, 4, 256, 2, 128, 1e-4],
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 8, 4, 256, 1, 128, 1e-4],


# retry after resume bugfix
# ["run_distributed_training.py", "--config-path conf --config-name train_lvae_dist_llnl", 8, 4, 256, 1, 128, 1e-4],
# debug bsz for single versus multilatent
# ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 1, 1, 64, 1, 128, 1e-4], # compile
# ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 1, 1, 128, 1, 128, 1e-4], # compile
# ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 1, 1, 256, 1, 128, 1e-4], # compile
# ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 1, 1, 512, 1, 128, 1e-4], # compile # overzealous
# ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 1, 1, 512, 1, 128, 1e-4], # compile # overzealous
# ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 2, 4, 256, 1, 128, 1e-4],
# ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 4, 4, 256, 1, 128, 1e-4],


# hit the pair as both 8N and 16N-ers?
# ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 8, 4, 256, 1, 128, 1e-4],
# ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 8, 4, 256, 1, 128, 1e-4],
# ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 16, 4, 256, 1, 128, 1e-4],
# ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 16, 4, 256, 1, 128, 1e-4],


# BASE_RUN_NAME = f"train_lvae_dist_debug_sweep"
# BASE_RUN_NAME = f"train_lvae_dist_prod"
# BASE_RUN_NAME = f"train_lvae_dist_crashtest_sweep"
 # ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 16, 4, 256, 1, 128, 1e-4],
# ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 16, 4, 256, 1, 128, 1e-4],
# crash testing
# ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 2, 4, 256, 1, 128, 1e-4],
# ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 4, 4, 256, 1, 128, 1e-4],
# ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 8, 4, 256, 1, 128, 1e-4],
# ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 16, 4, 256, 1, 128, 1e-4],
# ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 32, 4, 256, 1, 128, 1e-4],

# BASE_RUN_NAME = f"train_lvae_dist_restart_debugging"
# # ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 1, 1, 256, 1, 128, 1e-4, True],
# # ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 1, 1, 256, 1, 128, 1e-4, False],
# # ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 1, 4, 256, 1, 128, 1e-4, True],
# ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 1, 4, 256, 1, 128, 1e-4, False],
# # ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 2, 4, 256, 1, 128, 1e-4, True],
# ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 2, 4, 256, 1, 128, 1e-4, False],
# # ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 4, 4, 256, 1, 128, 1e-4, True],
# ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 4, 4, 256, 1, 128, 1e-4, False],