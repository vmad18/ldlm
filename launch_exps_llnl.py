# fmt: off
import os
from itertools import product, chain

# LIST_CFGS = True
LIST_CFGS = False

# WRITE_ONLY = True
WRITE_ONLY = False

LAUNCHER_FILEPATH = "/p/vast1/$USER/llnl-tools/launch_tuo.py"

RCCL_INSTALL_DIR = (
    "/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib"
)

ROCM_VERSION = "6.3.0"
RCCL_CFG = "rdzv-lbann"

# EXTRA_COMPILE_FLAGS = False
EXTRA_COMPILE_FLAGS = True

# LOG_RECOMPILES=False
LOG_RECOMPILES = True

QOS = "pdebug"
# QOS = "pbatch"

BANK = "guests"
# BANK = "effml"

TIME_LIMIT = 20
# TIME_LIMIT = 30
# TIME_LIMIT = 1440

REPETITIONS = 1
DEPENDENCY = None
# REPETITIONS = 3
# DEPENDENCY = "afterany"
# DEPENDENCY = "singleton"

BASE_OUT_DIR = f"/p/vast1/kirchenb/diffusion-root/ldlm/outputs"

# BASE_RUN_NAME = f"debug"
# BASE_RUN_NAME = f"compile_series"
BASE_RUN_NAME = f"scale_series"
# BASE_RUN_NAME = f"prod"

WANDB_OFFLINE = False
# WANDB_OFFLINE = True

# INVOCATION_PREAMBLE = "export UV_CACHE_DIR=$VASTUSER/.cache/uv && uv run --index-strategy=unsafe-best-match"
INVOCATION_PREAMBLE = "source .venv/bin/activate && python -u"

# MAX_STEPS = None
# # TGT_TOKENS = 100e9
# TGT_TOKENS = 300e9  # 100B tokens for 3 epochs

TGT_TOKENS = None
MAX_STEPS = 200

# flag to taggle special setup for chaining a compile warmup series
COMPILE_SERIES = False
# COMPILE_SERIES = True

if COMPILE_SERIES:
    assert (
        DEPENDENCY == "singleton"
    ), "Compile series warmup workflow requires singleton dependency"

# Cfgs
# gpn = gpus per node
# arg list is:
# script, cfg name, nodes, gpn, mbsz, accum, seq_len, lr ...
# exp_list = [
# ["run_distributed_training.py", "train_lvae_dist_llnl_multilat", 16, 4, 256, 1, 128, 1e-4, "True", None],
# ["run_distributed_training.py", "train_lvae_dist_llnl_singlelat", 16, 4, 256, 1, 128, 1e-4, "True", None],
# switching to single latents with smaller internal dims
# ]

ashwinee_cfgs = {
    "extreme_examples_1b": {
        "widest_model": {
            "d_model": 5120,
            "latent_dim": 384,
            "layers_p": 2,
            "params_millions": 992.02,
        },
        "narrowest_model": {
            "d_model": 256,
            "latent_dim": 2176,
            "layers_p": 20,
            "params_millions": 993.67,
        },
        "largest_latent": {
            "d_model": 256,
            "latent_dim": 7552,
            "layers_p": 2,
            "params_millions": 1003.84,
        },
        "smallest_latent": {
            "d_model": 2048,
            "latent_dim": 256,
            "layers_p": 18,
            "params_millions": 993.42,
        },
        "deepest_model": {
            "d_model": 384,
            "latent_dim": 1920,
            "layers_p": 24,
            "params_millions": 1002.09,
        },
        # "shallowest_model": {
        #     "d_model": 256,
        #     "latent_dim": 7552,
        #     "layers_p": 2,
        #     "params_millions": 1003.84,
        # },
    },
    "extreme_examples_2b": {
        "widest_model": {
            "d_model": 8192,
            "latent_dim": 5376,
            "layers_p": 2,
            "params_millions": 2487.57,
        },
        # "narrowest_model": {
        #     "d_model": 256,
        #     "latent_dim": 3328,
        #     "layers_p": 24,
        #     "params_millions": 2517.85,
        # },
        "largest_latent": {
            "d_model": 1280,
            "latent_dim": 8192,
            "layers_p": 4,
            "params_millions": 2486.47,
        },
        "smallest_latent": {
            "d_model": 3328,
            "latent_dim": 256,
            "layers_p": 21,
            "params_millions": 2521.47,
        },
        "deepest_model": {
            "d_model": 256,
            "latent_dim": 3328,
            "layers_p": 24,
            "params_millions": 2517.85,
        },
        "shallowest_model": {
            "d_model": 6400,
            "latent_dim": 8192,
            "layers_p": 2,
            "params_millions": 2505.12,
        },
    },
}

exp_list = [
    ["run_distributed_training.py", "train_lvae_dist_llnl", 1, 4, 1, 1, 128, 1e-4, 1e-4, "False", 1],
]

# sweep the model shapes
hparam_list = []
for cfg_name, models in ashwinee_cfgs.items():
    for model_name, model_cfg in models.items():
        d_model = model_cfg["d_model"]
        latent_dim = model_cfg["latent_dim"]
        layers_p = model_cfg["layers_p"]
        hparams = [
                d_model,
                latent_dim,
                layers_p,
            ]
        hparam_list.append(hparams)

exp_list = list(chain(*[[exp + hparams for hparams in hparam_list] for exp in exp_list]))

# then we will sweep the mbsz, and for the costliest model, use this to set the max wbsz that 8N or 16N allows
# then use only the node ct required for each less costly one


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
        kld,
        lr,
        compile_model,
        # lvae_path,
        num_lat,
        d_model,
        lat_dim,
        layers,
    ) = exp

    gpus = nodes * gpn

    cli_args = ""

    # config name
    cfg_name_str = cfg_name
    if "train_cfm_dist_llnl" in cfg_name:
        cfg_name = cfg_name.replace("_singlelat", "").replace("_multilat", "")
    cli_args += f" --config-path conf --config-name {cfg_name}"

    # mod lr and kld
    lr_name_str = f"lr{lr:.0e}-kl{kld:.0e}"
    lr_cfg_string = f" learning_rate={lr} kld_weight={kld}"
    cli_args += lr_cfg_string

    # mod bsz and seq len
    wbsz = nodes * gpn * mbsz * accum
    bsz_name_str = f"mb{mbsz}-acc{accum}-wb{wbsz}-seq{seq_len}"
    train_bsz_cfg_string = (
        f" train_bs={mbsz} grad_accumulate={accum} model.max_seq_len={seq_len}"
    )
    cli_args += train_bsz_cfg_string

    # mod shapes
    model_str = f"{num_lat}lat-{lat_dim}dlat-{d_model}dmod-{layers}lay"
    cli_args += (
        f" model.num_latents={num_lat} model.latent_dim={lat_dim} model.d_model={d_model} model.num_layers={layers}"
    )

    # compute max steps automatically for token target
    if MAX_STEPS is None and TGT_TOKENS is not None:
        max_steps = int(TGT_TOKENS / (wbsz * seq_len)) + 1
    elif MAX_STEPS is not None and TGT_TOKENS is None:
        max_steps = int(MAX_STEPS)
    else:
        raise ValueError(f"Either steps or toks control but not both")
    # compile series
    if COMPILE_SERIES:
        # shortcircuit the training
        cli_args += f" train_num_steps=10"
    else:
        # prod
        cli_args += f" train_num_steps={max_steps}"

    # compile model
    compile_str = "compiled" if compile_model else "uncompiled"
    cli_args += f" compile_model={compile_model}"

    # # add the lvae path
    # if lvae_path is not None:
    #     cli_args += f" model.lvae_model_path={lvae_path}"

    # mod more things
    # ...

    # join to a unique run name for the experiment
    # run_name = f"{cfg_name_str}_{nodes}N{gpus}n_{bsz_name_str}_{lr_name_str}_{compile_str}"
    # for compilation series
    if COMPILE_SERIES:
        # name such that they are an implicit slurm chain by sharing same name
        run_name = f"{cfg_name_str}"
    else:
        # prod
        run_name = (
            # f"{cfg_name_str}_{nodes}N{gpus}n_{bsz_name_str}_{lr_name_str}"
            # f"{cfg_name_str}_{nodes}N{gpus}n_{bsz_name_str}_{model_str}_{lr_name_str}"
            f"{BASE_RUN_NAME}_{model_str}_{bsz_name_str}_{nodes}N{gpus}n"
        )
    
    # add custom name for wandb
    cli_args += f" wandb_name={run_name}"
    cli_args += f" wandb_mode={'offline' if WANDB_OFFLINE else 'online'}"


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
        --wandb_offline={WANDB_OFFLINE} \
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
