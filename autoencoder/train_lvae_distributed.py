import os
import sys
import uuid
import time
import copy
import glob
import math
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache, partial
from collections import namedtuple, Counter
from contextlib import nullcontext
from typing import Tuple, Optional

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW

from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import wandb
from datetime import datetime, timedelta

# VAE imports - changed to absolute imports
from autoencoder.latent_vae import LatentVAEModel, get_latent_vae_tokenizer
from dataset_util.dataset_helper import get_dataloader_lvae_bin, wind_data_generator

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("Importing complete in train_lvae_distributed.py", flush=True)

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, and @vagrawal.
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            grad_pad = [param.grad for param in params] + [torch.zeros_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                # This gives strange dynamo warnings
                reduce_scatter_futures.append(dist.reduce_scatter(grad, grad_pad[base_i:base_i + world_size], op=dist.ReduceOp.AVG, async_op=True).get_future())

        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), world_size):
                reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad
                    eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                    eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    p.mul_(1 - eff_weight_decay)
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                all_reduce_futures.append(dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank], async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()

class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # DistributedAdam implementation by @vagrawal

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()

# Helper functions
def exists(x):
    return x is not None

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def separate_weight_decayable_params(params):
    # Exclude affine params in norms (e.g. LayerNorm, GroupNorm, etc.) and bias terms
    no_wd_params = [param for param in params if param.ndim < 2]
    wd_params = [param for param in params if param not in set(no_wd_params)]
    return wd_params, no_wd_params

def get_adamw_optimizer(params, lr, betas, weight_decay, eps=1e-8):
    params = list(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    param_groups = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]
    return AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)

def compute_grad_norm(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), p=2) for p in parameters]), p=2).item()
    return total_norm

# Gradient bucketing system (adapted from train_gpt.py)
def create_buckets(params, bucket_size_mb=25):
    """Group parameters into buckets of approximately bucket_size_mb MB each"""
    buckets = []
    current_bucket = []
    current_size = 0

    # Sort parameters by size (largest first) for better bucketing
    sorted_params = sorted(params, key=lambda p: p.numel(), reverse=True)

    for param in sorted_params:
        param_size_mb = param.numel() * param.element_size() / (1024 * 1024)

        if current_size + param_size_mb > bucket_size_mb and current_bucket:
            buckets.append(current_bucket)
            current_bucket = [param]
            current_size = param_size_mb
        else:
            current_bucket.append(param)
            current_size += param_size_mb

    if current_bucket:
        buckets.append(current_bucket)

    return buckets

def load_from_checkpoint(checkpoint_path, model, optimizer_adam, optimizer_muon, device, rank):
    """Load model and optimizers from external checkpoint directory."""
    checkpoint_path = Path(checkpoint_path)
    
    # Try to load from model_best.pt first, then model.pt
    checkpoint_file = None
    for filename in ['model_best.pt', 'model.pt']:
        potential_path = checkpoint_path / filename
        if potential_path.exists():
            checkpoint_file = potential_path
            break
    
    if checkpoint_file is None:
        raise FileNotFoundError(f"No checkpoint file (model_best.pt or model.pt) found in {checkpoint_path}")
    
    if rank == 0:
        print(f"Loading checkpoint from: {checkpoint_file}")
    
    # Load checkpoint data
    data = torch.load(str(checkpoint_file), map_location=device, weights_only=False)
    
    # Unwrap model if it's compiled or wrapped
    unwrapped_model = model
    if hasattr(model, '_orig_mod'):
        unwrapped_model = model._orig_mod
    
    # Load model state
    if 'model' in data:
        unwrapped_model.load_state_dict(data['model'])
        if rank == 0:
            print("Successfully loaded model weights from 'model' key.")
    else:
        # Handle case where checkpoint is just the state_dict
        unwrapped_model.load_state_dict(data)
        if rank == 0:
            print("Loaded model weights from raw state_dict.")
    
    # Initialize step and best_val_loss
    step = 0
    best_val_loss = float('inf')
    
    # Load training state
    if 'step' in data:
        step = data['step']
        if rank == 0:
            print(f"Resuming from step: {step}")
    
    optimizer_adam.load_state_dict(data['optimizer_adam'])
    optimizer_muon.load_state_dict(data['optimizer_muon'])
    optimizer_adam.zero_grad()
    optimizer_muon.zero_grad()
    if rank == 0:
        print("Loaded separate optimizer states.")
    
    if 'best_val_loss' in data:
        best_val_loss = data['best_val_loss']
        if rank == 0:
            print(f"Best validation loss: {best_val_loss}")
    
    # Load training time
    training_time_ms = data.get('training_time_ms', 0)
    
    # Restore RNG states for reproducibility
    if 'rng_state' in data:
        try:
            rng_state = data['rng_state']
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            print(f"rng_state['torch'].dtype: {rng_state['torch'].dtype}")
            print([state.dtype for state in rng_state['torch_cuda']])
            torch.set_rng_state(rng_state['torch'].cpu().byte())  # Should now be ByteTensor
            torch.cuda.set_rng_state_all([state.byte() for state in rng_state['torch_cuda']])  # Should now be list of ByteTensors
            if rank == 0:
                print("Restored RNG states for reproducibility.")
        except Exception as e:
            if rank == 0:
                print(e)
                print(f"Warning: Could not restore RNG states: {e}. Continuing with current RNG state.")
    
    if rank == 0:
        print("Checkpoint loading complete.")
    
    return step, best_val_loss, training_time_ms

def handle_checkpoint_config(cfg: DictConfig, rank: int):
    """Handle checkpoint loading and config merging (similar to train_lvae.py)."""
    # Handle checkpoint loading: if loading from external checkpoint, use its config for model creation
    if cfg.resume_from is not None:
        checkpoint_path = Path(cfg.resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
        
        config_path = checkpoint_path / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        if rank == 0:
            print(f"Loading config from checkpoint: {config_path}")
        
        # Load the checkpoint config and use its model config
        checkpoint_cfg = OmegaConf.load(config_path)
        
        # Use the model config from checkpoint, but keep other configs from current run
        if 'model' in checkpoint_cfg:
            cfg.model = checkpoint_cfg.model
        else:
            # If checkpoint config doesn't have model key, use the whole config as model config
            cfg.model = checkpoint_cfg
        
        if rank == 0:
            print("Using model architecture from checkpoint config")
    
    return cfg



def print0(s, logfile=None, console=False):
    """Print only from rank 0"""
    if dist.get_rank() == 0:
        if console:
            print(s)
        if logfile:
            with open(logfile, "a") as f:
                print(s, file=f)

def main(cfg: DictConfig):

    print(f"Top of main in train_lvae_distributed.py", flush=True)

    # Initialize distributed training
    # The convention is to:
    # 1. check the RANK, WORLD_SIZE, and LOCAL_RANK environment variables. launch_tuo.py and torchrun set these.
    # 2. if those aren't set, fall back to SLURM vars as this could still be run using vanilla srun in theory.
    # 3. else, set everything as if we are in single-GPU mode since it's not clear what's going on.
    rank = int(os.getenv("RANK", os.getenv("SLURM_PROCID", 0)))
    world_size = int(os.getenv("WORLD_SIZE", os.getenv("SLURM_NTASKS", 1)))
    local_rank = int(os.getenv("LOCAL_RANK")) if "LOCAL_RANK" in os.environ else rank % os.getenv("SLURM_NTASKS_PER_NODE", 1)
    master_process = (rank == 0)

    assert torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}")

    # Dist init is a synchronous operation that must execute properly.
    os.environ["TORCH_DIST_INIT_BARRIER"] = "1"  # ensure barrier is used for initialization

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,  # passing this immediately forms the NCCL communicator.
        timeout=timedelta(minutes=5),
    )
    # _Must_ this come after?
    # idk, @jog did it this way back in summer of 24' so that's good enough for me.
    torch.cuda.set_device(device)

    if master_process: print(f"{'#' * 80}", flush=True)
    dist.barrier()
    print(f"Distributed training initialized on rank ({rank}/{world_size}) as device={device} via {os.getenv("MASTER_ADDR", "null_badbad")}:{os.getenv("MASTER_PORT", "null_badbad")}",flush=True)
    dist.barrier()
    if master_process: print(f"{'#' * 80}", flush=True)

    # On APUs like MI300A and GH200, gpu and cpu memory are shared and so both types of allocations
    # fight for the same space. Tends to make things behave/fail better when you cap useable vram.
    if cfg.per_process_vram_ratio is not None:
        assert (
            cfg.per_process_vram_ratio > 0.0 and cfg.per_process_vram_ratio < 1.0
        ), f"Invalid per_process_vram_ratio: {cfg.per_process_vram_ratio}, must be in (0.0, 1.0)"
        torch.cuda.set_per_process_memory_fraction(cfg.per_process_vram_ratio)
        print(f"per_process_vram_ratio set to: {cfg.per_process_vram_ratio}")
    
    # Handle checkpoint config loading
    cfg = handle_checkpoint_config(cfg, rank)
    
    # Set seeds
    set_seeds(cfg.seed)
    
    # Setup logging
    logfile = None
    if master_process:
        cfg_hash = hashlib.md5(OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest()
        results_folder = Path(cfg.results_folder) if cfg.results_folder is not None else Path(cfg.output_dir) / cfg_hash
        
        # Check if we should auto-resume from existing checkpoint
        if cfg.resume_from is None and results_folder.exists():
            potential_checkpoint = results_folder / "model_best.pt"
            if potential_checkpoint.exists():
                cfg.resume_from = str(results_folder)
                print0(f"Auto-resuming from existing checkpoint: {cfg.resume_from}", console=True)
        
        results_folder.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(results_folder / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        
        # Setup wandb
        run_id = uuid.uuid4()
        logfile = results_folder / f"{run_id}.txt"
        
        wandb.init(
            project="lvae-distributed",
            name=f"{cfg.wandb_name}-{cfg_hash[:8]}" if cfg.wandb_name else f"run-{cfg_hash[:8]}",
            dir=str(results_folder),
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        
        print0(f"Results folder: {results_folder}", logfile, console=True)
        print0(f"Logfile: {logfile}", logfile, console=True)
    
    # Create model and tokenizer
    model, tokenizer = get_latent_vae_tokenizer(cfg.model)
    model = model.cuda()

    # Load checkpoint if resuming (AFTER compilation so optimizer refs are correct)
    step = 0
    best_val_loss = float('inf')
    training_time_ms = 0
    
    if cfg.resume_from is not None:
        if master_process:
            print0(f"Loading checkpoint from: {cfg.resume_from}", logfile, console=True)
        
        # Load model state first, then recreate optimizers to avoid parameter reference issues
        checkpoint_path = Path(cfg.resume_from)
        checkpoint_file = None
        for filename in ['model_best.pt', 'model.pt']:
            potential_path = checkpoint_path / filename
            if potential_path.exists():
                checkpoint_file = potential_path
                break
        
        if checkpoint_file is None:
            raise FileNotFoundError(f"No checkpoint file found in {checkpoint_path}")
        
        # Load checkpoint data
        data = torch.load(str(checkpoint_file), map_location=device, weights_only=False)
        
        # Load model state dict first (each rank loads the same state, so no broadcast needed)
        model.load_state_dict(data['model'])
        if master_process:
            print0("Loaded model state dict on all ranks", logfile, console=True)

    for param in model.parameters():
        dist.broadcast(param.detach(), 0)
    
    embed_params, head_params, pos_embed_params, hidden_matrix_params, scalar_params = [], [], [], [], []
    # Carefully separate parameters to avoid size mismatches
    for n, p in model.named_parameters():
        if "dembed_head" in n:
            head_params.append(p)
            if master_process:
                print(f"Head param: {n}, Shape: {p.shape}")
        elif "pos_embed" in n:
            pos_embed_params.append(p)
            if master_process:
                print(f"Pos embed param: {n}, Shape: {p.shape}")
        elif "embed" in n:
            embed_params.append(p)
            if master_process:
                print(f"Embed param: {n}, Shape: {p.shape}")
        elif p.ndim >= 2:
            hidden_matrix_params.append(p)
            if master_process:
                print(f"Hidden matrix param: {n}, Shape: {p.shape}")
        elif p.ndim < 2:
            scalar_params.append(p)
            if master_process:
                print(f"Scalar param: {n}, Shape: {p.shape}")

    optimizer_adam = DistAdam(scalar_params + head_params + embed_params, lr=cfg.learning_rate, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
    optimizer_muon = Muon(hidden_matrix_params + pos_embed_params, lr=cfg.muon_lr, momentum=0.95, weight_decay=0.0)
    
    # Set initial_lr for proper learning rate scheduling after resume
    for group in optimizer_adam.param_groups:
        group['initial_lr'] = cfg.learning_rate
    for group in optimizer_muon.param_groups:
        group['initial_lr'] = cfg.muon_lr
    
    optimizers = [optimizer_adam, optimizer_muon]
    
    # Verify no parameter overlap between optimizers
    muon_param_ids = {id(p) for group in optimizer_muon.param_groups for p in group['params']}
    adam_param_ids = {id(p) for group in optimizer_adam.param_groups for p in group['params']}
    overlap = muon_param_ids & adam_param_ids
    
    if overlap:
        raise ValueError(f"Found {len(overlap)} overlapping parameters between Adam and Muon optimizers during checkpoint loading!")

    if cfg.resume_from is not None:
        optimizer_adam.load_state_dict(data['optimizer_adam'])
        optimizer_muon.load_state_dict(data['optimizer_muon'])
        
        if master_process:
            print0("Recreated and loaded optimizer states", logfile, console=True)
        
        # Load other checkpoint data
        step = data['step']
        best_val_loss = data['best_val_loss']
        training_time_ms = data['training_time_ms']
        
        if master_process:
            print0(f"Checkpoint loaded successfully. Resuming from step {step}", logfile, console=True)
    
    # Setup data loaders AFTER checkpoint loading so we can wind to correct position
    if cfg.resume_from is not None and step > 0:
        # We're resuming training, so wind the data generator to the correct position
        if master_process:
            print0(f"Winding data generator to step {step}", logfile, console=True)
        
        # Wind the training data generator
        train_loader = wind_data_generator(
            cfg,
            cfg.train_bin_pattern,
            step,
            rank,
            world_size,
            tokenizer=tokenizer
        )
    else:
        # Normal startup, create fresh data loaders
        train_loader = get_dataloader_lvae_bin(
            cfg, 
            cfg.train_bin_pattern, 
            rank, 
            world_size,
            tokenizer=tokenizer
        )
        
    val_loader = get_dataloader_lvae_bin(
        cfg, 
        cfg.val_bin_pattern, 
        rank, 
        world_size,
        tokenizer=tokenizer
    )
    
    # Setup learning rate scheduling
    def get_lr(step: int):
        # Simple linear warmup then cosine decay
        warmup_steps = 1000
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (cfg.train_num_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    def get_annealed_kld_weight(step: int):
        """Compute the current KLD weight with linear annealing."""
        if step < cfg.kld_annealing_steps:
            return cfg.kld_weight * (step / cfg.kld_annealing_steps)
        else:
            return cfg.kld_weight

    if master_process:
        print0("Starting training...", logfile, console=True)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    first_step = True
    for step in range(step, cfg.train_num_steps + 1):
        last_step = (step == cfg.train_num_steps)
        
        # Validation
        if last_step or (cfg.eval_every > 0 and step % cfg.eval_every == 0):
            # Stop training timer
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            
            model.eval()
            val_loss = 0
            val_recon_loss = 0
            val_kld_loss = 0
            num_val_batches = 25
            
            with torch.no_grad():
                for _ in range(num_val_batches):
                    batch = next(val_loader)
                    batch = {k: v.cuda() for k, v in batch.items()}
                    
                    losses = model(batch["input_ids"], attn_mask=batch["attention_mask"])
                    current_kld_weight = get_annealed_kld_weight(step)
                    loss = losses['reconstruction_loss'] + current_kld_weight * losses['kld_loss']
                    
                    val_loss += loss.item()
                    val_recon_loss += losses['reconstruction_loss'].item()
                    val_kld_loss += losses['kld_loss'].item()
            
            val_loss /= num_val_batches
            val_recon_loss /= num_val_batches
            val_kld_loss /= num_val_batches
            
            # Reduce validation losses across all ranks
            val_loss_tensor = torch.tensor(val_loss, device=device)
            val_recon_loss_tensor = torch.tensor(val_recon_loss, device=device)
            val_kld_loss_tensor = torch.tensor(val_kld_loss, device=device)
            
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_recon_loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_kld_loss_tensor, op=dist.ReduceOp.AVG)
            
            val_loss = val_loss_tensor.item()
            val_recon_loss = val_recon_loss_tensor.item()
            val_kld_loss = val_kld_loss_tensor.item()
            
            if master_process:
                # Calculate progress
                tokens_processed = step * cfg.train_bs * cfg.grad_accumulate * world_size * cfg.model.max_seq_len
                epoch = tokens_processed / cfg.total_tokens
                
                print0(f"step:{step}/{cfg.train_num_steps} val_loss:{val_loss:.4f} "
                       f"val_recon_loss:{val_recon_loss:.4f} val_kld_loss:{val_kld_loss:.4f} "
                    #    f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms",
                       f"train_time:{training_time_ms/1e3:.0f}s step_avg:{training_time_ms/1e3/max(step, 1):.2f}s",
                       logfile, console=True)
                
                # Log to wandb
                logs = {
                    "val/loss": val_loss,
                    "val/reconstruction_loss": val_recon_loss,
                    "val/kld_loss": val_kld_loss,
                    "step": step,
                    "epoch": epoch,
                    "training_time_ms": training_time_ms
                }
                
                # Save checkpoint
                if cfg.save_checkpoint:
                    # Get the unwrapped model for saving
                    unwrapped_model = model
                    if hasattr(model, '_orig_mod'):
                        unwrapped_model = model._orig_mod
                    
                    checkpoint = {
                        'step': step,
                        'model': unwrapped_model.state_dict(),  # Save unwrapped model state
                        'optimizer_adam': optimizer_adam.state_dict(),
                        'optimizer_muon': optimizer_muon.state_dict(),
                        'best_val_loss': best_val_loss,
                        'val_loss': val_loss,
                        'training_time_ms': training_time_ms,
                        'rng_state': {
                            'python': random.getstate(),
                            'numpy': np.random.get_state(),
                            'torch': torch.get_rng_state(),  # Ensure CPU tensor
                            'torch_cuda': torch.cuda.get_rng_state_all(),  # Ensure CPU tensors
                        }
                    }
                    
                    torch.save(checkpoint, results_folder / "model.pt")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(checkpoint, results_folder / "model_best.pt")
                        logs['val/best_loss'] = best_val_loss
                        print(f"New best val loss: {best_val_loss}")
                
                # Generate samples for logging
                if step % (cfg.eval_every * 2) == 0:
                    # Reconstruction samples
                    batch = next(val_loader)
                    batch = {k: v.cuda() for k, v in batch.items()}
                    input_ids = batch["input_ids"][:4]
                    attn_mask = batch["attention_mask"][:4]
                    
                    embeddings = model.embed(input_ids)
                    recon_embeds, _, _ = model.vae(embeddings, attn_mask.bool())
                    recon_logits = model.dembed_head(recon_embeds[..., :input_ids.shape[1], :])
                    recon_ids = torch.argmax(recon_logits, dim=-1)
                    
                    original_texts = tokenizer.batch_decode(input_ids.cpu(), skip_special_tokens=True)
                    reconstructed_texts = tokenizer.batch_decode(recon_ids.cpu(), skip_special_tokens=True)
                    
                    recon_table = wandb.Table(columns=["Original", "Reconstructed"])
                    for orig, recon in zip(original_texts, reconstructed_texts):
                        recon_table.add_data(orig, recon)
                    logs["reconstructions"] = recon_table
                    
                    # Generation samples
                    num_gen_samples = 4
                    latents = torch.randn((num_gen_samples, model.num_latents, model.latent_dim), device=device)
                    gen_logits = model.decode_latent(latents)
                    gen_ids = torch.argmax(gen_logits, dim=-1)
                    generated_texts = tokenizer.batch_decode(gen_ids.cpu(), skip_special_tokens=True)
                    
                    gen_table = wandb.Table(columns=["Generated Text"])
                    for gen in generated_texts:
                        gen_table.add_data(gen)
                    logs["generated_samples"] = gen_table
                
                wandb.log(logs, step=step)
            
            model.train()
            
            # Restart training timer
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        
        if last_step:
            break
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0
        
        for _ in range(cfg.grad_accumulate):
            batch = next(train_loader)
            batch = {k: v.cuda() for k, v in batch.items()}
            
            losses = model(batch["input_ids"], attn_mask=batch["attention_mask"])
            
            recon_loss = losses['reconstruction_loss']
            kld_loss = losses['kld_loss']
            current_kld_weight = get_annealed_kld_weight(step)
            loss = (recon_loss + current_kld_weight * kld_loss) / cfg.grad_accumulate
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item() / cfg.grad_accumulate
            total_kld_loss += kld_loss.item() / cfg.grad_accumulate
            
            loss.backward()
        
        grad_norm = compute_grad_norm(model.parameters())
        # Update learning rates
        current_lr = get_lr(step)
        
        # Adam optimizer - use cfg.learning_rate as base
        for group in optimizer_adam.param_groups:
            group["lr"] = group.get("initial_lr", cfg.learning_rate) * current_lr
        
        # Muon optimizer - use cfg.muon_lr as base  
        for group in optimizer_muon.param_groups:
            group["lr"] = group.get("initial_lr", cfg.muon_lr) * current_lr
        
        # Momentum warmup for Muon (only if momentum isn't already warmed up)
        if step < 300:
            frac = min(step / 300, 1)
            for group in optimizer_muon.param_groups:
                group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        
        # Optimizer step
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        
        # Logging
        if step % cfg.log_step_interval == 0 or first_step:
            tokens_processed = step * cfg.train_bs * cfg.grad_accumulate * world_size * cfg.model.max_seq_len
            epoch = tokens_processed / cfg.total_tokens
            
            approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)

            time_remaining = (cfg.train_num_steps - step) * (approx_training_time_ms / (step + 1))
            # time_remaining_str = str(timedelta(milliseconds=int(time_remaining)))
            time_remaining_hrs = int(time_remaining) / 1e3 / 3600
            
            print0(f"step:{step+1}/{cfg.train_num_steps} loss:{total_loss:.4f} "
                   f"recon_loss:{total_recon_loss:.4f} kld_loss:{total_kld_loss:.4f} "
                   f"kld_weight:{current_kld_weight:.6f} grad_norm:{grad_norm:.4f} "
                   f"lr:{current_lr:.6f} "
                #    f"train_time:{approx_training_time_ms:.0f}ms "
                #    f"step_avg:{approx_training_time_ms/(step + 1):.2f}ms",
                    f"train_time:{approx_training_time_ms/1e3:.0f}s "
                    f"step_avg:{approx_training_time_ms/1e3/(step + 1):.2f}s "
                    f"epoch:{epoch:.4f} tokens:{tokens_processed} "
                    f"time_remaining_hrs:{time_remaining_hrs:.2f}hr",
                   logfile, console=True)
            if master_process:
                # if step % 100 == 0:
                logs = {
                    "train/loss": total_loss,
                    "train/reconstruction_loss": total_recon_loss,
                    "train/kld_loss": total_kld_loss,
                    "train/kld_weight": current_kld_weight,
                    "train/grad_norm": grad_norm,
                    "train/lr": current_lr,
                    "step": step,
                    "epoch": epoch,
                    "samples": tokens_processed,
                    "time_remaining_hrs": time_remaining_hrs,
                }
                wandb.log(logs, step=step)
        first_step = False
    # Final logging
    if master_process:
        print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
               f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", logfile, console=True)
        print0('Training complete', logfile, console=True)
    
    dist.destroy_process_group()
    return best_val_loss