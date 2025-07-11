# fmt: off
import os
from itertools import product, chain

# LIST_CFGS = True
LIST_CFGS = False

# WRITE_ONLY = True
WRITE_ONLY = False

LAUNCHER_FILEPATH = "/p/vast1/$USER/llnl-tools/launch_tuo.py"

RCCL_INSTALL_DIR = "/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib"

ROCM_VERSION = "6.3.0"
RCCL_CFG = "rdzv-lbann"

# WANDB_OFFLINE = "False"
WANDB_OFFLINE = "True"

# QOS = "pdebug"
QOS = "pbatch"

# BANK = "guests"
BANK = "effml"

# TIME_LIMIT = 10  # in minutes
TIME_LIMIT = 1440 

REPETITIONS = 1
# REPETITIONS = 3
# DEPENDENCY = "afterany"
DEPENDENCY = None

BASE_OUT_DIR = f"/p/vast1/kirchenb/diffusion-root/ldlm/outputs"

# BASE_RUN_NAME = f"test_sweep"
# BASE_RUN_NAME = f"train_lvae_100b_debug"
BASE_RUN_NAME = f"train_lvae_100b"

INVOCATION_PREAMBLE = "export UV_CACHE_DIR=$VASTUSER/.cache/uv && uv run --index-strategy=unsafe-best-match"

# Cfgs
# gpn = gpus per node
# arg list is:
# script, cli_args, nodes, gpn, mbsz, accum, seq_len, lr, ...
exp_list = [
    # ["main_lvae.py", "", 1, 1, 48, 16, 512, 1e-4],
    # ["main_lvae.py", "", 1, 1, 64, 12, 512, 1e-4],
    ["main_lvae.py", "", 1, 1, 96, 8, 512, 1e-4],
    # ["main_lvae.py", "", 1, 1, 128, 6, 512, 1e-4],
]

# add an additional sweep for each prev cfg over somthing like lr or seed etc.
# sweep_hparam = [
#     ["null"],
#     [1234, 4321, 1738],
# ]
# exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))

final_exp_list = exp_list
for exp in final_exp_list:
    print(exp)

total_launches = 0

# queue all jobs
for exp in final_exp_list:

    (
        script,
        cli_args,
        nodes,
        gpn,
        mbsz,
        accum,
        seq_len,
        lr,
    ) = exp

    gpus = nodes * gpn

    # mod lr
    lr_name_str = f"lr{lr:.0e}"
    lr_cfg_string = f" training.optimizer.learning_rate={lr}"
    cli_args += lr_cfg_string

    # mod bsz and seq len
    wbsz = nodes * gpn * mbsz * accum
    bsz_name_str = f"mb{mbsz}-acc{accum}-wb{wbsz}-seq{seq_len}"
    train_bsz_cfg_string = f" training.train_bs={mbsz} training.grad_accumulate={accum} model.max_seq_len={seq_len}"
    cli_args += train_bsz_cfg_string

    # mod more things 
    # ...

    # join to a unique run name for the experiment
    run_name = f"{BASE_RUN_NAME}_{script.strip('.py')}_{nodes}N{gpus}n_{bsz_name_str}_{lr_name_str}"

    # put together the actual "train.py" command
    custom_invocation = f"{INVOCATION_PREAMBLE} {script} {cli_args}"

    # make the complete launcher command
    command = f"""\
    python {LAUNCHER_FILEPATH} \
        --output_dir={BASE_OUT_DIR}/{BASE_RUN_NAME} \
        --rocm_version={ROCM_VERSION} \
        --rccl_installdir={RCCL_INSTALL_DIR} \
        --rccl_cfg={RCCL_CFG} \
        --qos={QOS} \
        --bank={BANK} \
        --repetitions={REPETITIONS}{f' --dependency={DEPENDENCY}' if DEPENDENCY is not None else ''} \
        --minutes={TIME_LIMIT} \
        --nodes={nodes} \
        --gpus_per_node={gpn} \
        --run_name={run_name} \
        --custom_invocation='{custom_invocation}' \
        --pass_run_name=False \
        {'--dryrun' if WRITE_ONLY else ''}
    """
    total_launches += 1
    if not LIST_CFGS:
        os.system(command)
    else:
        print(run_name)
        # print(command)

print(f"Total launches: {total_launches}")
