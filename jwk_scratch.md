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
