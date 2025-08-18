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

# QOS = "pdebug"
QOS = "pbatch"

# BANK = "guests"
BANK = "effml"

TIME_LIMIT = 29

REPETITIONS = 1
DEPENDENCY = None

BASE_OUT_DIR = f"/p/vast1/kirchenb/diffusion-root/ldlm/outputs"

# BASE_RUN_NAME = f"debug"
BASE_RUN_NAME = f"scale_series_nodes_vs_mbsz"

WANDB_OFFLINE = False
# WANDB_OFFLINE = True

INVOCATION_PREAMBLE = "source .venv/bin/activate && python -u"

# INDUCTOR_CACHE=None
INDUCTOR_CACHE="/l/ssd/$USER"

# MAX_STEPS = None
# # TGT_TOKENS = 100e9
# TGT_TOKENS = 300e9  # 100B tokens for 3 epochs

TGT_TOKENS = None
MAX_STEPS = 100

TOK_WBSZ_1M = 8192 * 128
TOK_WBSZ_4M = TOK_WBSZ_1M * 4

SEQ_LEN = 128

GPN = 4

MAX_MEM = None
# MAX_MEM = 0.9

# flag to taggle special setup for chaining a compile warmup series
COMPILE_SERIES = False
# COMPILE_SERIES = True

if COMPILE_SERIES:
    assert (
        DEPENDENCY == "singleton"
    ), "Compile series warmup workflow requires singleton dependency"

# Cfgs
ashwinee_cfgs = {
    "orig_single_lat": {
        "reference": {
            "d_model": 768,
            "latent_dim": 2048,
            "layers_p": 12,
        },
    },
    "extreme_examples_1b": {
        "widest_model": {
            "d_model": 5120,
            "latent_dim": 384,
            "layers_p": 2,
        },
        "narrowest_model": {
            "d_model": 256,
            "latent_dim": 2176,
            "layers_p": 20,
        },
        "largest_latent": {
            "d_model": 256,
            "latent_dim": 7552,
            "layers_p": 2,
        },
        "smallest_latent": {
            "d_model": 2048,
            "latent_dim": 256,
            "layers_p": 18,
        },
        "deepest_model": {
            "d_model": 384,
            "latent_dim": 1920,
            "layers_p": 24,
        },
    },
    "extreme_examples_2b": {
        "widest_model": {
            "d_model": 8192,
            "latent_dim": 5376,
            "layers_p": 2,
        },
        # # seem like could be too big to run even at the min bsz
        # "largest_latent": {
        #     "d_model": 1280,
        #     "latent_dim": 8192,
        #     "layers_p": 4,
        # },
        "smallest_latent": {
            "d_model": 3328,
            "latent_dim": 256,
            "layers_p": 21,
        },
        "deepest_model": {
            "d_model": 256,
            "latent_dim": 3328,
            "layers_p": 24,
        },
        "shallowest_model": {
            "d_model": 6400,
            "latent_dim": 8192,
            "layers_p": 2,
        },
    },
}

exp_list = [
    ["run_distributed_training.py", "train_lvae_dist_llnl", 1e-4, 1e-4, "True", 1, 1],
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

hparam_list = [1,2,4,8,16,32]
exp_list = list(chain(*[[exp + [hparam] for hparam in hparam_list] for exp in exp_list]))

hparam_list = [1,2,4,8,16,32,64,128,256,512]
exp_list = list(chain(*[[exp + [hparam] for hparam in hparam_list] for exp in exp_list]))

final_exp_list = exp_list
for exp in final_exp_list:
    print(exp)

total_launches = 0

# queue all jobs
for exp in final_exp_list:

    (
        script,
        cfg_name,
        kld,
        lr,
        compile_model,
        accum,
        num_lat,
        d_model,
        lat_dim,
        layers,
        nodes,
        mbsz,
    ) = exp

    gpn = GPN
    gpus = nodes * GPN
    seq_len = SEQ_LEN

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

    if MAX_MEM is not None:
        cli_args += f" per_process_vram_ratio={MAX_MEM}"

    # mod more things
    # ...

    # join to a unique run name for the experiment
    # for compilation series
    if COMPILE_SERIES:
        # name such that they are an implicit slurm chain by sharing same name
        run_name = f"{cfg_name_str}"
    else:
        # prod
        run_name = (
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
        --cache_dir={INDUCTOR_CACHE} \
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
        # print(command)

print(f"Total launches: {total_launches}")
