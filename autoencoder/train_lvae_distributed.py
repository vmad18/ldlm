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
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW

from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import wandb
from datetime import datetime

# VAE imports - changed to absolute imports
from autoencoder.latent_vae import LatentVAEModel, get_latent_vae_tokenizer
from dataset_util.dataset_helper import get_dataloader_lvae_bin, wind_data_generator

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class TrainingConfig:
    # Model configuration
    model_config: DictConfig  # Changed from model to model_config
    
    # Training configuration
    train_num_steps: int = 10000
    train_bs: int = 8
    eval_bs: int = 8
    eval_every: int = 1000
    grad_accumulate: int = 1
    
    # Data configuration
    train_bin_pattern: str = "data/train_*.bin"
    val_bin_pattern: str = "data/val_*.bin"
    total_tokens: int = 100_000_000_000  # 100B tokens
    
    # Optimization configuration
    learning_rate: float = 1e-4
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    adam_weight_decay: float = 0.1
    adam_eps: float = 1e-8
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    
    # VAE configuration
    kld_weight: float = 1e-4
    kld_annealing_steps: int = 2000
    
    # Logging configuration
    seed: int = 42
    output_dir: str = "outputs"
    wandb_name: Optional[str] = None
    save_checkpoint: bool = True
    
    # Resume configuration
    resume_from: Optional[str] = None

# Muon optimizer (copied from train_gpt.py)
@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[torch.Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: torch.Tensor = group["update_buffer"]
            update_buffer_views: list[torch.Tensor] = group["update_buffer_views"]
            params: list[torch.Tensor] = group["params"]
            handle = None
            params_world = None
            
            def update_prev():
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
                    
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: torch.Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

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
    
    if 'opt' in data:
        # For backwards compatibility - single optimizer
        optimizer_adam.load_state_dict(data['opt'])
        if rank == 0:
            print("Loaded optimizer state from 'opt' key.")
    elif 'optimizer_adam' in data and 'optimizer_muon' in data:
        # New format - separate optimizers
        optimizer_adam.load_state_dict(data['optimizer_adam'])
        optimizer_muon.load_state_dict(data['optimizer_muon'])
        if rank == 0:
            print("Loaded separate optimizer states.")
    elif 'optimizer_adam' in data:
        optimizer_adam.load_state_dict(data['optimizer_adam'])
        if rank == 0:
            print("Loaded Adam optimizer state.")
    
    if 'best_val_loss' in data:
        best_val_loss = data['best_val_loss']
        if rank == 0:
            print(f"Best validation loss: {best_val_loss}")
    
    if rank == 0:
        print("Checkpoint loading complete.")
    
    return step, best_val_loss

def handle_checkpoint_config(cfg: TrainingConfig, rank: int):
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
            cfg.model_config = checkpoint_cfg.model  # Changed to model_config
        else:
            # If checkpoint config doesn't have model key, use the whole config as model config
            cfg.model_config = checkpoint_cfg  # Changed to model_config
        
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

def main(cfg: TrainingConfig):
    print(f"Top of main in train_lvae_distributed.py", flush=True)

    # Initialize distributed training
    # The convention is to:
    # 1. check the RANK, WORLD_SIZE, and LOCAL_RANK environment variables. launch_tuo.py and torchrun set these.
    # 2. if those aren't set, fall back to SLURM vars as this could still be run using vanilla srun in theory.
    # 3. else, set everything as if we are in single-GPU mode since it's not clear what's going on.
    rank = int(os.getenv("RANK", os.getenv("SLURM_PROCID", 0)))
    world_size = int(os.getenv("WORLD_SIZE", os.getenv("SLURM_NTASKS", 1)))
    master_process = (rank == 0)
    # assert torch.cuda.is_available()
    local_rank = int(os.getenv("LOCAL_RANK")) if "LOCAL_RANK" in os.environ else rank % os.getenv("SLURM_NTASKS_PER_NODE", 1)
    device = torch.device("cuda", local_rank)

    print(f"About to initialize distributed on rank ({rank}/{world_size}) as device={device}")
    
    # first synchronous operation that must execute properly
    # dist.init_process_group(backend="nccl", device_id=device)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,  # this immediately forms the NCCL communicator
    )
    # Must this come after?
    torch.cuda.set_device(device)

    # pre-barrier log each ranks setup
    if dist.is_initialized():
        print(f"Distributed training initialized on rank ({rank}/{world_size}) as device={device}")
    # wait for all ranks
    # dist.barrier()
    torch.distributed.barrier() # barrier not barriering?
    # log "success" message (still possible topo is misconfigured but no errors yet)
    print0("Distributed training setup successful. All ranks initialized correctly.", console=True)

    # On APUs like MI300A and GH200, gpu and cpu memory are shared and so both types of allocations
    # fight for the same space. Tends to make things behave/fail better when you cap useable vram.
    # if cfg.per_process_vram_ratio is not None:
    #     assert (
    #         cfg.per_process_vram_ratio > 0.0 and cfg.per_process_vram_ratio < 1.0
    #     ), f"Invalid per_process_vram_ratio: {cfg.per_process_vram_ratio}, must be in (0.0, 1.0)"
    #     torch.cuda.set_per_process_memory_fraction(cfg.per_process_vram_ratio)
    #     print(f"per_process_vram_ratio set to: {cfg.per_process_vram_ratio}")
    
    # Handle checkpoint config loading
    cfg = handle_checkpoint_config(cfg, rank)
    
    # Set seeds
    set_seeds(cfg.seed)
    
    # Setup logging
    logfile = None
    if master_process:
        # Create unique directory for this run
        cfg_str = OmegaConf.to_yaml(cfg.model_config, resolve=True)  # Changed to model_config
        cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()
        results_folder = Path(cfg.output_dir) / cfg_hash
        
        # Check if we should auto-resume from existing checkpoint
        if cfg.resume_from is None and results_folder.exists():
            potential_checkpoint = results_folder / "model_best.pt"
            if potential_checkpoint.exists():
                cfg.resume_from = str(results_folder)
                print0(f"Auto-resuming from existing checkpoint: {cfg.resume_from}", console=True)
        
        results_folder.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(results_folder / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg.model_config, resolve=True))  # Changed to model_config
        
        # Setup wandb
        run_id = uuid.uuid4()
        logfile = results_folder / f"{run_id}.txt"
        
        wandb.init(
            project="lvae-distributed",
            name=f"{cfg.wandb_name}-{cfg_hash[:8]}" if cfg.wandb_name else f"run-{cfg_hash[:8]}",
            dir=str(results_folder),
            config=OmegaConf.to_container(cfg.model_config, resolve=True)  # Changed to model_config
        )
        
        print0(f"Results folder: {results_folder}", logfile, console=True)
        print0(f"Logfile: {logfile}", logfile, console=True)
    
    # Create model and tokenizer
    model, tokenizer = get_latent_vae_tokenizer(cfg.model_config)  # Changed to model_config
    model = model.cuda()
    
    # Broadcast model parameters to all ranks
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)
    
    # Separate parameters for different optimizers
    # Hidden matrix params (2D parameters) → Muon optimizer
    hidden_matrix_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    # Embedding params → Adam
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    # Scalar params → Adam
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    # Head params → Adam (dembed_head)
    head_params = [p for n, p in model.named_parameters() if "dembed_head" in n]
    
    # Initialize optimizers
    adam_params = [
        dict(params=head_params, lr=cfg.learning_rate),
        dict(params=embed_params, lr=cfg.learning_rate),
        dict(params=scalar_params, lr=cfg.learning_rate)
    ]
    
    optimizer_adam = get_adamw_optimizer(
        [p for group in adam_params for p in group['params']],
        lr=cfg.learning_rate,
        betas=cfg.adam_betas,
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_eps
    )
    
    optimizer_muon = Muon(
        hidden_matrix_params,
        lr=cfg.muon_lr,
        momentum=cfg.muon_momentum,
        rank=rank,
        world_size=world_size
    )
    
    optimizers = [optimizer_adam, optimizer_muon]
    
    # Setup gradient bucketing
    all_params = [p for p in model.parameters() if p.requires_grad]
    param_buckets = create_buckets(all_params)
    
    if master_process:
        print0(f"Created {len(param_buckets)} gradient buckets", logfile, console=True)
        for i, bucket in enumerate(param_buckets):
            total_size = sum(p.numel() * p.element_size() for p in bucket) / (1024 * 1024)
            print0(f"Bucket {i}: {len(bucket)} params, {total_size:.1f} MB", logfile, console=True)
    
    # Bucket state tracking
    bucket_ready_count = [0] * len(param_buckets)
    bucket_handles = [None] * len(param_buckets)
    param_to_bucket = {}
    
    # Map each parameter to its bucket index
    for bucket_idx, bucket in enumerate(param_buckets):
        for param in bucket:
            param_to_bucket[param] = bucket_idx
    
    def _gradient_hook(param: torch.Tensor):
        """Called when a parameter's gradient is ready"""
        if param.grad is None:
            return
        
        bucket_idx = param_to_bucket[param]
        bucket_ready_count[bucket_idx] += 1
        
        # Check if all parameters in this bucket are ready
        if bucket_ready_count[bucket_idx] == len(param_buckets[bucket_idx]):
            # All-reduce this bucket
            bucket_grads = [p.grad for p in param_buckets[bucket_idx]]
            
            if len(bucket_grads) == 1:
                handle = dist.all_reduce(bucket_grads[0], op=dist.ReduceOp.AVG, async_op=True)
            else:
                handle = dist.all_reduce_coalesced(bucket_grads, op=dist.ReduceOp.AVG, async_op=True)
            
            bucket_handles[bucket_idx] = handle
    
    # Register hooks for all parameters
    if master_process:
        print0("Registering bucketed gradient hooks...", logfile, console=True)
    
    for param in all_params:
        param.register_post_accumulate_grad_hook(_gradient_hook)
    
    def wait_for_gradients():
        """Wait for all gradient reductions to complete and reset bucket state"""
        for handle in bucket_handles:
            if handle is not None:
                handle.wait()
        
        # Reset state for next iteration
        for i in range(len(bucket_ready_count)):
            bucket_ready_count[i] = 0
            bucket_handles[i] = None
    
    # Load checkpoint if resuming
    step = 0
    best_val_loss = float('inf')
    
    if cfg.resume_from is not None:
        if master_process:
            print0(f"Loading checkpoint from: {cfg.resume_from}", logfile, console=True)
        
        step, best_val_loss = load_from_checkpoint(
            cfg.resume_from, model, optimizer_adam, optimizer_muon, device, rank
        )
    
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
        
        # For validation, we don't need to wind (always start from beginning)
        val_loader = get_dataloader_lvae_bin(
            cfg, 
            cfg.val_bin_pattern, 
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
    
    # Compile model
    model = torch.compile(model, dynamic=False)
    
    # Warmup kernels - but skip if we're resuming from a checkpoint
    if cfg.resume_from is None:
        if master_process:
            print0("Warming up kernels...", logfile, console=True)
        
        warmup_steps = 3
        initial_state = dict(
            model=copy.deepcopy(model.state_dict()),
            optimizer_adam=copy.deepcopy(optimizer_adam.state_dict()),
            optimizer_muon=copy.deepcopy(optimizer_muon.state_dict())
        )
        
        for _ in range(warmup_steps):
            batch = next(train_loader)
            batch = {k: v.cuda() for k, v in batch.items()}
            
            losses = model(batch["input_ids"], attn_mask=batch["attention_mask"])
            loss = losses['reconstruction_loss'] + get_annealed_kld_weight(0) * losses['kld_loss']
            loss.backward()
            
            wait_for_gradients()
            
            for opt in optimizers:
                opt.step()
            
            model.zero_grad(set_to_none=True)
        
        # Restore initial state
        model.load_state_dict(initial_state["model"])
        optimizer_adam.load_state_dict(initial_state["optimizer_adam"])
        optimizer_muon.load_state_dict(initial_state["optimizer_muon"])
        del initial_state
    else:
        if master_process:
            print0("Skipping kernel warmup (resuming from checkpoint)", logfile, console=True)
    
    # Training loop
    if master_process:
        print0("Starting training...", logfile, console=True)
    
    training_time_ms = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
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
                tokens_processed = step * cfg.train_bs * cfg.grad_accumulate * world_size * cfg.model_config.max_seq_len  # Changed to model_config
                epoch = tokens_processed / cfg.total_tokens
                
                print0(f"step:{step}/{cfg.train_num_steps} val_loss:{val_loss:.4f} "
                       f"val_recon_loss:{val_recon_loss:.4f} val_kld_loss:{val_kld_loss:.4f} "
                       f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms",
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
                        'val_loss': val_loss
                    }
                    
                    torch.save(checkpoint, results_folder / "model.pt")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(checkpoint, results_folder / "model_best.pt")
                        logs['val/best_loss'] = best_val_loss
                
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
        
        # Training step
        model.zero_grad(set_to_none=True)
        
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
        
        # Wait for all gradients to be reduced
        wait_for_gradients()
        
        # Gradient clipping
        grad_norm = compute_grad_norm(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update learning rates
        current_lr = get_lr(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group.get("initial_lr", cfg.learning_rate) * current_lr
        
        # Momentum warmup for Muon
        if step < 300:
            frac = min(step / 300, 1)
            for group in optimizer_muon.param_groups:
                group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        
        # Optimizer step
        for opt in optimizers:
            opt.step()
        
        # Logging
        if master_process and step % 50 == 0:
            tokens_processed = step * cfg.train_bs * cfg.grad_accumulate * world_size * cfg.model_config.max_seq_len  # Changed to model_config
            epoch = tokens_processed / cfg.total_tokens
            
            approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
            
            print0(f"step:{step+1}/{cfg.train_num_steps} loss:{total_loss:.4f} "
                   f"recon_loss:{total_recon_loss:.4f} kld_loss:{total_kld_loss:.4f} "
                   f"kld_weight:{current_kld_weight:.6f} grad_norm:{grad_norm:.4f} "
                   f"lr:{current_lr:.6f} train_time:{approx_training_time_ms:.0f}ms "
                   f"step_avg:{approx_training_time_ms/(step + 1):.2f}ms",
                   logfile, console=True)
            
            if step % 100 == 0:
                logs = {
                    "train/loss": total_loss,
                    "train/reconstruction_loss": total_recon_loss,
                    "train/kld_loss": total_kld_loss,
                    "train/kld_weight": current_kld_weight,
                    "train/grad_norm": grad_norm,
                    "train/lr": current_lr,
                    "step": step,
                    "epoch": epoch,
                    "samples": tokens_processed
                }
                wandb.log(logs, step=step)
    
    # Final logging
    if master_process:
        print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
               f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", logfile, console=True)
        print0('Training complete', logfile, console=True)
    
    dist.destroy_process_group()
    return best_val_loss