# first whack at setup on LLNL

Install script seems to be workable

not sure that conda->uv, with .venv inside the repo is optimal but pin for now

with base conda up, on a Tuo compute node in -N1 -n1 allocation
after intial path hacking runs where the dataset was actually pulled, we have:
```
uv run --index-strategy=unsafe-best-match main_lvae.py
...
num trainable params: 707495425
Loading pre-split dataset from /p/vast1/kirchenb/.cache/ldlm/datasets/fineweb_edu_splits/fineweb_edu_split_valid10
Loading dataset from disk: 100%|██████████████████████████████████████████████████████████████| 87/87 [00:00<00:00, 365.13it/s]
Loading dataset from disk: 100%|█████████████████████████████████████████████████████████████| 64/64 [00:00<00:00, 9093.03it/s]
Loading tokenized dataset from disk: /p/vast1/kirchenb/.cache/ldlm/tokenized_ds/fineweb-edu_10b/meta-llama__Llama-3.2-1B_seqlen128
Loading tokenized dataset from disk: /p/vast1/kirchenb/.cache/ldlm/tokenized_ds/fineweb-edu_10b/meta-llama__Llama-3.2-1B_seqlen128
Starting comprehensive warm-up pass to compile forward and backward graphs...
Warm-up pass complete.
  2%|▋                                       | 160/10000 [05:47<5:52:44,  2.15s/it, kld_loss=0.517, loss=7.21, recon_loss=7.21]
```


basic demo of how we might launch

conda_activate $VASTUSER/tuolumne_uv_ldlm && \ 
python /p/lustre5/$USER/llnl-tools/launch_tuo.py \
    --rocm_version=6.3.0 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pbatch \
    --bank=effml \
    --repetitions=1 \
    --minutes=1440 \
    --nodes=1 \
    --gpus_per_node=1 \
    --output_dir=/p/vast1/kirchenb/diffusion-root/ldlm/outputs \
    --run_name=babys_first_lvae_N1n1 \
    --pass_run_name=False \
    --custom_invocation='export UV_CACHE_DIR=$VASTUSER/.cache/uv && uv run --index-strategy=unsafe-best-match main_lvae.py' \
&& \
python /p/lustre5/$USER/llnl-tools/launch_tuo.py \
    --rocm_version=6.3.0 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pbatch \
    --bank=effml \
    --repetitions=1 \
    --minutes=1440 \
    --nodes=1 \
    --gpus_per_node=4 \
    --output_dir=/p/vast1/kirchenb/diffusion-root/ldlm/outputs \
    --run_name=babys_first_lvae_N1n4 \
    --pass_run_name=False \
    --custom_invocation='export UV_CACHE_DIR=$VASTUSER/.cache/uv && uv run --index-strategy=unsafe-best-match main_lvae.py'


Next steps are:
- launching automation using llnl-tools since the slurm hydra stuff probably wont work
- testing 1N4n, but not really sure whether code has ever been run in multigpu

# see whether the diffusion training part works too

python /p/lustre5/$USER/llnl-tools/launch_tuo.py \
    --rocm_version=6.3.0 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pbatch \
    --bank=effml \
    --repetitions=1 \
    --minutes=1440 \
    --nodes=1 \
    --gpus_per_node=1 \
    --output_dir=/p/vast1/kirchenb/diffusion-root/ldlm/outputs \
    --run_name=babys_first_ldlm_N1n1 \
    --pass_run_name=False \
    --custom_invocation='export UV_CACHE_DIR=$VASTUSER/.cache/uv && uv run main_diff.py model.latent_model_path=/p/vast1/kirchenb/diffusion-root/ldlm/outputs/2025-07-08/21-12-18/ff5a4f56ed6c9067e6196c93d0127cdb'

pointing that at a ckpt from the prev ongoing N1n1 run gives something like:
```
0: 0: Loading VAE from path: /p/vast1/kirchenb/diffusion-root/ldlm/outputs/2025-07-08/21-12-18/ff5a4f56ed6c9067e6196c93d0127cdb
0: 0: Loaded VAE and its tokenizer (meta-llama/Llama-3.2-1B).
0: 0: 
  0%|          | 0/80000 [00:00<?, ?it/s]0: 0: Loading VAE checkpoint from: /p/vast1/kirchenb/diffusion-root/ldlm/outputs/2025-07-08/21-12-18/ff5a4f56ed6c9067e6196c93d0127cdb/model_best.pt
0: 0: Successfully loaded VAE model weights.
0: 0: Creating ConditionalFlowMatcher
0: 0: Loading pre-split dataset from /p/vast1/kirchenb/.cache/ldlm/datasets/fineweb_edu_splits/fineweb_edu_split_valid10
0: 0: Loading tokenized dataset from disk: /p/vast1/kirchenb/.cache/ldlm/tokenized_ds/fineweb-edu_10b/meta-llama__Llama-3.2-1B_seqlen128
0: 0: Loading tokenized dataset from disk: /p/vast1/kirchenb/.cache/ldlm/tokenized_ds/fineweb-edu_10b/meta-llama__Llama-3.2-1B_seqlen128
0: 0: Model params: 2560.56 M
0: 0: 
  0%|          | 0/80000 [00:38<?, ?it/s, epoch=0, grad_norm=0.105, learning_rate=1e-7, loss=1.63, samples=0, step=0, val_ema_loss=nan, val_loss=nan]0: 
  0%|          | 1/80000 [00:38<850:24:41, 38.27s/it, epoch=0, grad_norm=0.105, learning_rate=1e-7, loss=1.63, samples=0, step=0, val_ema_loss=nan, val_loss=nan]0: ...
0: 0: 
  0%|          | 12/80000 [01:24<98:40:19,  4.44s/it, epoch=0.00132, grad_norm=0.105, learning_rate=1.2e-6, loss=1.63, samples=11264, step=11, val_ema_loss=nan, val_loss=nan] 
```

# taking a look at correctness of multigpu impl

python /p/lustre5/$USER/llnl-tools/launch_tuo.py \
    --rocm_version=6.3.0 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pbatch \
    --bank=guests \
    --repetitions=1 \
    --minutes=240 \
    --nodes=1 \
    --gpus_per_node=4 \
    --output_dir=/p/vast1/kirchenb/diffusion-root/ldlm/outputs \
    --run_name=multigpu_debug_N1n4 \
    --pass_run_name=False \
    --custom_invocation='export UV_CACHE_DIR=$VASTUSER/.cache/uv && uv run --index-strategy=unsafe-best-match main_lvae.py' \
    --dryrun


# get the big data


python /p/lustre5/$USER/llnl-tools/launch_tuo.py \
  --rocm_version=6.3.0 \
  --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib \
  --rccl_cfg=rdzv-lbann \
  --qos=pbatch \
  --bank=effml \
  --repetitions=1 \
  --minutes=1440 \
  --nodes=1 \
  --gpus_per_node=1 \
  --output_dir=/p/vast1/kirchenb/diffusion-root/ldlm/outputs \
  --run_name=tok_fw_350b_N1n1 \
  --pass_run_name=False \
  --custom_invocation='export UV_CACHE_DIR=$VASTUSER/.cache/uv && uv run --index-strategy=unsafe-best-match main_lvae.py data.dataset_name=fineweb_350b'


# restarting the existing lvae job

conda_activate $VASTUSER/tuolumne_uv_ldlm && \
python /p/lustre5/$USER/llnl-tools/launch_tuo.py \
    --rocm_version=6.3.0 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pbatch \
    --bank=effml \
    --repetitions=1 \
    --minutes=1440 \
    --nodes=1 \
    --gpus_per_node=1 \
    --output_dir=/p/vast1/kirchenb/diffusion-root/ldlm/outputs \
    --run_name=babys_first_lvae_N1n1 \
    --pass_run_name=False \
    --custom_invocation='export UV_CACHE_DIR=$VASTUSER/.cache/uv && uv run --index-strategy=unsafe-best-match main_lvae.py model.latent_model_path=/p/vast1/kirchenb/diffusion-root/ldlm/outputs/2025-07-08/21-12-24/ff5a4f56ed6c9067e6196c93d0127cdb'


# test the merged in bin file data logic

conda_activate $VASTUSER/tuolumne_uv_ldlm && \
python /p/lustre5/$USER/llnl-tools/launch_tuo.py \
    --rocm_version=6.3.0 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pdebug \
    --bank=effml \
    --repetitions=1 \
    --minutes=59 \
    --nodes=1 \
    --gpus_per_node=1 \
    --output_dir=/p/vast1/kirchenb/diffusion-root/ldlm/outputs \
    --run_name=debug_bin_logic_lvae_N1n1 \
    --pass_run_name=False \
    --custom_invocation='export UV_CACHE_DIR=$VASTUSER/.cache/uv && uv run --index-strategy=unsafe-best-match main_lvae.py'

(cfm can only be run with a vae trained on the correct tokenizer)

python /p/lustre5/$USER/llnl-tools/launch_tuo.py \
    --rocm_version=6.3.0 \
    --rccl_installdir=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-6.3.1/install/lib \
    --rccl_cfg=rdzv-lbann \
    --qos=pdebug \
    --bank=effml \
    --repetitions=1 \
    --minutes=59 \
    --nodes=1 \
    --gpus_per_node=1 \
    --output_dir=/p/vast1/kirchenb/diffusion-root/ldlm/outputs \
    --run_name=debug_bin_logic_ldlm_N1n1 \
    --pass_run_name=False \
    --custom_invocation='export UV_CACHE_DIR=$VASTUSER/.cache/uv && uv run main_diff.py model.latent_model_path=/p/vast1/kirchenb/diffusion-root/ldlm/outputs/2025-07-11/11-01-22/d3ca981bc3e2e1a6ce7bdc396dd21939'

Okay I think we're good on the new data (at least for N1n1).

# next, tune the hparams to make it go brr