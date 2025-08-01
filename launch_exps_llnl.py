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

# EXTRA_COMPILE_FLAGS = False
EXTRA_COMPILE_FLAGS = True

# LOG_RECOMPILES=False
LOG_RECOMPILES=True

# QOS = "pdebug"
QOS = "pbatch"

# BANK = "guests"
BANK = "effml"

# TIME_LIMIT = 15
# TIME_LIMIT = 30
TIME_LIMIT = 1440 

# REPETITIONS = 1
# DEPENDENCY = None
REPETITIONS = 3
DEPENDENCY = "afterany"
# DEPENDENCY = "singleton"

BASE_OUT_DIR = f"/p/vast1/kirchenb/diffusion-root/ldlm/outputs"

BASE_RUN_NAME = f"prod"
# BASE_RUN_NAME = f"compile_series"
# BASE_RUN_NAME = f"compile_series_w_compile_model"
# BASE_RUN_NAME = f"compile_series_w_10m_timeout"

# INVOCATION_PREAMBLE = "export UV_CACHE_DIR=$VASTUSER/.cache/uv && uv run --index-strategy=unsafe-best-match"
INVOCATION_PREAMBLE = "source .venv/bin/activate && python -u"

# TGT_TOKENS = 100e9
TGT_TOKENS = 300e9  # 100B tokens for 3 epochs

# flag to taggle special setup for chaining a compile warmup series
COMPILE_SERIES = False
# COMPILE_SERIES = True

if COMPILE_SERIES:
    assert DEPENDENCY == "singleton", "Compile series warmup workflow requires singleton dependency"

# Cfgs
# gpn = gpus per node
# arg list is:
# script, cfg name, nodes, gpn, mbsz, accum, seq_len, lr ...
exp_list = [
    # ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 1, 4, 256, 1, 128, 1e-4],
    # ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 1, 4, 256, 1, 128, 1e-4],
    # ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 2, 4, 256, 1, 128, 1e-4],
    # ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 2, 4, 256, 1, 128, 1e-4],
    # ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 4, 4, 256, 1, 128, 1e-4],
    # ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 4, 4, 256, 1, 128, 1e-4],
    # ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 8, 4, 256, 1, 128, 1e-4],
    # ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 8, 4, 256, 1, 128, 1e-4],
    ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 16, 4, 256, 1, 128, 1e-4],
    ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 16, 4, 256, 1, 128, 1e-4],
]

final_exp_list = exp_list
for exp in final_exp_list:
    print(exp)

total_launches = 0

# queue all jobs
for exp in final_exp_list:

    (
        script,
        cfg_name,
        nodes,
        gpn,
        mbsz,
        accum,
        seq_len,
        lr,
        # compile_model,
    ) = exp

    gpus = nodes * gpn

    cli_args = ""

    # config name
    cfg_name_str = cfg_name
    cli_args += f" --config-path conf --config-name {cfg_name}"

    # mod lr
    lr_name_str = f"lr{lr:.0e}"
    lr_cfg_string = f" learning_rate={lr}"
    cli_args += lr_cfg_string

    # mod bsz and seq len
    wbsz = nodes * gpn * mbsz * accum
    bsz_name_str = f"mb{mbsz}-acc{accum}-wb{wbsz}-seq{seq_len}"
    train_bsz_cfg_string = f" train_bs={mbsz} grad_accumulate={accum} model.max_seq_len={seq_len}"
    cli_args += train_bsz_cfg_string

    # compute max steps automatically for token target
    max_steps = int(TGT_TOKENS / (wbsz*seq_len))+1
    # compile series
    if COMPILE_SERIES:
        # shortcircuit the training
        cli_args += f" train_num_steps=10"
    else:
        # prod
        cli_args += f" train_num_steps={max_steps}"

    # compile model
    # compile_str = "compiled" if compile_model else "uncompiled"
    # cli_args += f" compile_model={compile_model}"


    # mod more things 
    # ...

    # join to a unique run name for the experiment
    # run_name = f"{cfg_name}_{nodes}N{gpus}n_{bsz_name_str}_{lr_name_str}_{compile_str}"
    # for compilation series
    if COMPILE_SERIES:
        # name such that they are an implicit slurm chain by sharing same name
        run_name = f"{cfg_name}"
    else:
        # prod
        run_name = f"{cfg_name}_{nodes}N{gpus}n_{bsz_name_str}_{lr_name_str}"

    # last thing, add our manual result dir
    res_folder = f"{BASE_OUT_DIR}/{BASE_RUN_NAME}/{run_name}"
    cli_args += f" results_folder={res_folder}"

    # put together the actual "train.py" command
    custom_invocation = f"{INVOCATION_PREAMBLE} {script} {cli_args}"
    
    # for compilation series
    if COMPILE_SERIES:
        # clear the prev ckpts
        custom_invocation = f"rm -rf {res_folder}/*.pt && {custom_invocation}"
    else:
        # prod
        pass

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
        --add_compile_flags={EXTRA_COMPILE_FLAGS} \
        --log_recompiles={LOG_RECOMPILES} \
        {'--dryrun' if WRITE_ONLY else ''}
    """
    total_launches += 1
    if not LIST_CFGS:
        os.system(command)
    else:
        print(run_name)
        print(command)

print(f"Total launches: {total_launches}")
