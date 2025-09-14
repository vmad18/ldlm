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
# BANK = "guests"
# TIME_LIMIT = 29

QOS = "pbatch"
BANK = "effml"
# BANK = "guard"
TIME_LIMIT = 1440

# REPETITIONS = 1
# DEPENDENCY = None
REPETITIONS = 5
DEPENDENCY = "afterany"

BASE_OUT_DIR = f"/p/vast1/kirchenb/diffusion-root/ldlm/outputs"

# BASE_RUN_NAME = f"prod_slss_ds_abl"
BASE_RUN_NAME = f"prod_slss_long"

WANDB_OFFLINE = False
# WANDB_OFFLINE = True

INVOCATION_PREAMBLE = "source .venv/bin/activate && python -u"

# INDUCTOR_CACHE=None
INDUCTOR_CACHE="/l/ssd/$USER"

# MAX_STEPS = None
# TGT_TOKENS = 100e9
# TGT_TOKENS = 300e9  # 100B tokens for 3 epochs
# TGT_TOKENS = 1e12

TGT_TOKENS = None
# MAX_STEPS = 100e3
MAX_STEPS = 500e3

TOK_WBSZ_1M = 8192 * 128
TOK_WBSZ_4M = TOK_WBSZ_1M * 4
MAX_NODES = 32

SEQ_LEN = 128

GPN = 4


MAX_MEM = None
# MAX_MEM = 0.9

# Cfgs
ashwinee_cfgs = {
    "orig_single_lat": {
        "reference": {
            "d_model": 768,
            "latent_dim": 2048,
            "layers_p": 12,
            "max_mbsz": 256,
            "max_node_ct": 8,
            "accum_for_tgt": 1, 
            "tgt_tok_wbsz": TOK_WBSZ_1M,
        },
    },
}

exp_list = [
    ["run_distributed_training.py", "train_lvae_dist_llnl", "True", 1],
]

# sweep the model shapes
hparam_list = []
for cfg_name, models in ashwinee_cfgs.items():
    for model_name, model_cfg in models.items():
        print(model_cfg)
        d_model = model_cfg["d_model"]
        latent_dim = model_cfg["latent_dim"]
        layers_p = model_cfg["layers_p"]
        hparams = [
                d_model,
                latent_dim,
                layers_p,
            ]
        # compute the multiplier of the mbsz and seq_len that would be required to hit the tgt tok wbsz
        
        tgt_tok_wbsz = model_cfg["tgt_tok_wbsz"]
        specd_max_mbsz = model_cfg["max_mbsz"]
        specd_accum = model_cfg["accum_for_tgt"]
        specd_max_node_ct = model_cfg["max_node_ct"]

        assert specd_max_node_ct * GPN * specd_max_mbsz * specd_accum * SEQ_LEN == tgt_tok_wbsz
        assert specd_max_node_ct <= MAX_NODES
        
        # add the hparams
        hparams += [
            specd_max_node_ct,
            GPN,
            specd_max_mbsz,
            specd_accum,
        ]

        hparam_list.append(hparams)

exp_list = list(chain(*[[exp + hparams for hparams in hparam_list] for exp in exp_list]))

# orig hparams
# lr, muon_lr, kld
# 1e-4, 2e-2, 1e-4

# lr
sweep_hparam = [
   6e-5,
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))

# muon_lr
sweep_hparam = [
   7e-3,
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))

# kld
sweep_hparam = [
#    1e-5,
   1e-4,
]
exp_list = list(chain(*[[exp + [hp] for hp in sweep_hparam] for exp in exp_list]))


# data
sweep_hparam = [
    # [
    # "/p/vast1/kirchenb/.cache/ldlm/datasets/fineweb100B/fineweb_train_*.bin",
    # "/p/vast1/kirchenb/.cache/ldlm/datasets/fineweb100B/fineweb_val_*.bin",
    # ],
    # [
    # "/p/vast1/kirchenb/.cache/ldlm/binary_datasets/rocstories_gpt2/roc_train_*.bin",
    # "/p/vast1/kirchenb/.cache/ldlm/binary_datasets/rocstories_gpt2/roc_train_*.bin",
    # # "/p/vast1/kirchenb/.cache/ldlm/binary_datasets/rocstories_gpt2/roc_test_*.bin",
    # 4111142, # toks in train
    # ],
    [
    "/p/vast1/kirchenb/.cache/ldlm/binary_datasets/tinystories_gpt2/tiny_train_*.bin",
    "/p/vast1/kirchenb/.cache/ldlm/binary_datasets/tinystories_gpt2/tiny_validation_*.bin",
    473992006, # toks in train
    ],
]
exp_list = list(chain(*[[exp + hp for hp in sweep_hparam] for exp in exp_list]))


final_exp_list = exp_list
for exp in final_exp_list:
    print(exp)

total_launches = 0

# queue all jobs
for exp in final_exp_list:

    (
        script,
        cfg_name,
        compile_model,
        num_lat,
        d_model,
        lat_dim,
        layers,
        nodes,
        gpn,
        mbsz,
        accum,
        lr,
        muon_lr,
        kld,
        tr_pattern,
        val_pattern,
        toks_in_tr,
        # lvae_path, # will be auto selecting from final runname
    ) = exp

    gpus = nodes * GPN
    seq_len = SEQ_LEN

    cli_args = ""

    # config name
    cfg_name_str = cfg_name
    if "train_cfm_dist_llnl" in cfg_name:
        cfg_name = cfg_name.replace("_singlelat", "").replace("_multilat", "")
    cli_args += f" --config-path conf --config-name {cfg_name}"

    # mod lr and kld
    lr_name_str = f"lr{lr:.0e}-mlr{muon_lr:.0e}-kld{kld:.0e}"
    lr_cfg_string = f" learning_rate={lr} muon_lr={muon_lr} kld_weight={kld}"
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

    cli_args += f" train_num_steps={max_steps}"

    # compile model
    compile_str = "compiled" if compile_model else "uncompiled"
    cli_args += f" compile_model={compile_model}"

    if MAX_MEM is not None:
        cli_args += f" per_process_vram_ratio={MAX_MEM}"

    # dataset stuff
    if "fineweb" in tr_pattern:
        ds_str = "fw100b-ds"
    elif "rocstories" in tr_pattern:
        ds_str = "roc-ds"
    elif "tinystories" in tr_pattern:
        ds_str = "tiny-ds"
    
    # cli_args += f" train_bin_pattern={tr_pattern} val_bin_pattern={val_pattern}"
    cli_args += f" train_bin_pattern={tr_pattern} val_bin_pattern={val_pattern} total_tokens={toks_in_tr}"

    # mod more things
    # ...

    # join to a unique run name for the experiment
    run_name = (
        # f"{BASE_RUN_NAME}_{ds_str}_{lr_name_str}_{model_str}_{bsz_name_str}_{nodes}N{gpus}n"
        f"{BASE_RUN_NAME}_{ds_str}_{bsz_name_str}_{nodes}N{gpus}n"
    )

    # # add the lvae path
    # if lvae_path is not None:
    #     cli_args += f" model.lvae_model_path={lvae_path}"
    
    # add custom name for wandb
    cli_args += f" wandb_name={run_name}"
    cli_args += f" wandb_mode={'offline' if WANDB_OFFLINE else 'online'}"


    # last thing, add our manual result dir
    res_folder = f"{BASE_OUT_DIR}/{BASE_RUN_NAME}/{run_name}"
    cli_args += f" results_folder={res_folder}"

    # put together the actual "train.py" command
    custom_invocation = f"{INVOCATION_PREAMBLE} {script} {cli_args}"

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
