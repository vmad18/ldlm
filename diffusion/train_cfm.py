from argparse import Namespace
from datetime import timedelta

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
from ema_pytorch import EMA

from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerBase, T5ForConditionalGeneration, \
    MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from autoencoder.latent_vae import get_latent_vae_tokenizer
from dataset_util.dataset_helper import get_dataset, get_dataloader_lvae, get_dataloader_lvae_bin, get_val_dataloader_lvae_bin
from omegaconf import DictConfig, OmegaConf

from diffusion.cond_flow_matcher import ConditionalFlowMatcher
from diffusion.neural_diffusion import DiTModel, DiTConfig

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
import wandb

import os
import hashlib

import copy

from typing import Optional, Tuple
import torch.distributed as dist

from pathlib import Path
from datetime import datetime
import json
import random

__version__ = "0.0.1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_output_dir(args):
    model_dir = f'{Path(args.dataset_name).stem}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    output_dir = os.path.join(args.save_dir, model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Created {output_dir}')
    return output_dir


def cycle(dl):
    while True:
        for data in dl:
            yield data


def exists(x):
    return x is not None


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


def euler_solver(
                x_0: torch.Tensor, 
                t_steps: torch.Tensor, 
                model: DiTModel, 
                pbar=None):
    x = x_0
    dt = t_steps[1] - t_steps[0]

    for t_step in range(t_steps.shape[0] - 1):
        t = t_steps[t_step] * torch.ones(x_0.shape[0], device=x_0.device)
        t = ConditionalFlowMatcher.pad_t_like_x(t, x)
        dx = model(x, t, x_0)  # approx. trajectory step
        # print(dx - x_0)
        x = x + dx * dt  # euler integrate

        if pbar is not None:
            pbar.update(1)
    return x


def gen_samples(
        model: DiTModel,
        num_latents: int,
        dim_latents: int,
        batch_size: int,
        accelerator: Accelerator,
        steps: int,  # number of gen steps
        target_dtype: torch.dtype,
        method: str = "euler"):
    # The caller of this function should handle model.eval() and model.train()
    with torch.no_grad():
        x_0 = torch.randn((batch_size, num_latents, dim_latents), device=accelerator.device, dtype=target_dtype)
        t_steps = torch.linspace(0, 1, steps + 1, device=accelerator.device)

        if method == "euler":
            traj = euler_solver(x_0, t_steps, model, None)
        else:
            raise NotImplementedError
    return traj


def compute_grad_norm(parameters):
    # implementation adapted from https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), p=2) for p in parameters]), p=2).item()
    return total_norm


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


def ema(source: nn.Module, target: nn.Module, decay: float):
    """ ema update """
    source_dict = source.state_dict()
    target_dict = target.state_dict()

    for key in source_dict.keys():
        target_dict[key].data.copy_(source_dict[key].data * decay + target_dict[key].data * (1 - decay))


class Trainer(object):

    @classmethod
    def from_pretrained_for_generation(
                                cls, 
                                checkpoint_dir: str, 
                                mixed_precision: str = "bf16"):
        """
        Loads a pre-trained model from a checkpoint directory for generation.
        This is a factory method that constructs a trainer in "generation mode"
        without needing to instantiate any of the training-specific components.
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # --- 1. Load Training Args from Checkpoint ---
        # Be specific to avoid loading the vae.pt file by mistake
        model_files = list(checkpoint_dir.glob('model-*.pt')) + list(checkpoint_dir.glob('model.pt'))
        if not model_files:
            raise FileNotFoundError(f"No diffusion model checkpoint found in {checkpoint_dir}")
        
        model_checkpoint_path = max(model_files, key=os.path.getctime)
        print(f"Loading diffusion model checkpoint from: {model_checkpoint_path}")
        data = torch.load(model_checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'cfg' not in data:
            raise ValueError(f"Checkpoint {model_checkpoint_path} does not contain 'cfg'. Cannot restore model.")
        args = data['cfg']

        # --- 2. Create a bare Trainer instance and Accelerator ---
        trainer = cls.__new__(cls)
        trainer.accelerator = Accelerator(mixed_precision=mixed_precision)
        
        assert trainer.accelerator.num_processes == 1, "Multi-gpu training is no bueno rn"
        print(
            f"Accelerator (RANK: {trainer.accelerator.process_index}, "
            f"LOCAL_RANK: {trainer.accelerator.local_process_index}, "
            f"WORLD_SIZE: {trainer.accelerator.num_processes}) - "
            f"Mixed Precision: {trainer.accelerator.mixed_precision}, "
            f"Device: {trainer.accelerator.device}, "
        )
        # these have no setters and are not constructor args (straight to jail)
        # self.accelerator.local_process_index = os.getenv("LOCAL_RANK", 0)
        # self.accelerator.process_index = os.getenv("RANK", 0)
        # self.accelerator.num_processes = os.getenv("WOLRD_SIZE", 1)
        # so we're gonna ignore them

        # --- 3. Manually Set Up VAE and Tokenizer ---
        vae_args_path = checkpoint_dir / 'vae_args.json'
        if not vae_args_path.exists():
            raise FileNotFoundError(f"VAE args not found at {vae_args_path}. Checkpoint directory may be incomplete.")
        with open(vae_args_path, 'r') as f:
            trainer.vae_args = json.load(f)
        
        trainer.tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp", use_fast=False)
        vae_model_path = checkpoint_dir / 'vae.pt'
        if not vae_model_path.exists():
            raise FileNotFoundError(f"VAE model not found at {vae_model_path}. Checkpoint directory may be incomplete.")
        trainer.ae = torch.load(vae_model_path, map_location='cpu', weights_only=False)
        trainer.ae.to(trainer.accelerator.device)
        trainer.ae.eval()

        # --- 4. Manually Set Up Diffusion Model ---
        trainer.num_latents = trainer.ae.num_latents
        trainer.latent_dim = trainer.ae.latent_dim
        
        cfg_dit = DiTConfig()
        cfg_dit.dim = args.model_dim
        cfg_dit.num_latents = trainer.num_latents
        cfg_dit.latent_dim = trainer.latent_dim
        cfg_dit.num_layers = args.num_layers
        cfg_dit.dev = trainer.accelerator.device
        
        trainer.v_predictor = DiTModel(cfg_dit)
        trainer.ema_model = copy.deepcopy(trainer.v_predictor)

        # --- 5. Prepare Models and Set Generation Attributes ---
        trainer.v_predictor, trainer.ema_model = trainer.accelerator.prepare(trainer.v_predictor, trainer.ema_model)
        trainer.output_dir = checkpoint_dir
        trainer.step = 0 # Will be updated by load()

        if trainer.accelerator.mixed_precision == "fp16":
            trainer.model_dtype = torch.float16
        elif trainer.accelerator.mixed_precision == "bf16":
            trainer.model_dtype = torch.bfloat16
        else:
            trainer.model_dtype = torch.float32
        
        # --- 6. Load the Model Weights ---
        trainer.load_for_generation(model_checkpoint_path)
        return trainer

    def __init__(self,
                 cfg: DictConfig,
                 output_dir: str = "./results"
                 ) -> None:
        # This __init__ method is now EXCLUSIVELY for training.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=90))

        self.accelerator = Accelerator(
            mixed_precision=cfg.training.mixed_precision,
            gradient_accumulation_steps=cfg.training.grad_accumulate,
            log_with="wandb",
            kwargs_handlers=[ddp_kwargs, init_process_kwargs],
        )

        # Determine the target dtype based on the mixed precision setting
        if self.accelerator.mixed_precision == "fp16":
            self.model_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.model_dtype = torch.bfloat16
        else:
            self.model_dtype = torch.float32

        self.num_dev = self.accelerator.num_processes
        self.cfg = cfg

        if self.accelerator.is_main_process:
            # Create a unique directory for each run based on its config hash
            cfg_str = OmegaConf.to_yaml(cfg, resolve=True)
            cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()
            self.output_dir = Path(output_dir) / cfg_hash
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # save the config to the results folder
            with open(self.output_dir / "config.yaml", "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))
            print(f"Results folder for this run: {self.output_dir}")
        else:
            # For non-main processes, we still need to set output_dir for consistency
            cfg_str = OmegaConf.to_yaml(cfg, resolve=True)
            cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()
            self.output_dir = Path(output_dir) / cfg_hash

        self.save_and_sample_every = cfg.training.save_and_sample_every
        self.train_bs = cfg.training.train_bs
        self.eval_bs = cfg.training.eval_bs
        self.gradient_accumulate_every = cfg.training.grad_accumulate
        self.ema_update_every = cfg.training.ema_update_every
        self.eval_every = cfg.training.eval_every

        # generation args from eval config
        self.gen_steps = cfg.eval.gen_steps
        self.gen_max_length = cfg.eval.gen_max_length
        self.num_gen_samples = cfg.eval.num_gen_samples
        self.gen_batch_size = cfg.eval.gen_batch_size
        
        print(f"Loading VAE from path: {cfg.model.latent_model_path}")
        vae_config_path = os.path.join(cfg.model.latent_model_path, 'config.yaml')
        if not os.path.exists(vae_config_path):
            raise FileNotFoundError(f"VAE config not found at {vae_config_path}")
        
        loaded_vae_cfg = OmegaConf.load(vae_config_path)

        # Instantiate VAE and its corresponding tokenizer from the saved config
        vae_model_config = loaded_vae_cfg.model
        
        # TODO: potentially a temporary patch to handle loading prev run cfgs that didnt spec tokenizer
        if not hasattr(loaded_vae_cfg.model, "tokenizer_name"):
            assert hasattr(cfg.model, "tokenizer_name") is not None, (
            "If the loaded_vae_cfg.model doesn't specify a tokenizer, the current cfg.model needs to spec one."
            "Proceed with caution, technically leaves room for a mismatch."
            )
            print(f"Loading tokenizer according to runtime config (not loaded VAE ckpt).")
            vae_model_config.tokenizer_name = cfg.model.tokenizer_name

        self.ae, self.tokenizer = get_latent_vae_tokenizer(vae_model_config)
        print(f"Loaded VAE and its tokenizer ({self.tokenizer.name_or_path}).")
        
        # Move model to device before loading state_dict
        self.ae.to(self.accelerator.device)

        vae_checkpoint_path = os.path.join(cfg.model.latent_model_path, 'model_best.pt')
        if not os.path.exists(vae_checkpoint_path):
            raise FileNotFoundError(f"VAE checkpoint 'model_best.pt' not found in {cfg.model.latent_model_path}")

        print(f"Loading VAE checkpoint from: {vae_checkpoint_path}")
        vae_data = torch.load(vae_checkpoint_path, map_location=self.accelerator.device, weights_only=False)
        
        if 'model' in vae_data:
            self.ae.load_state_dict(vae_data['model'])
            print("Successfully loaded VAE model weights.")
        else:
            self.ae.load_state_dict(vae_data)
            print("Loaded VAE weights from raw state_dict.")

        del vae_data
        
        self.ae.eval()
        for param in self.ae.parameters():
            param.requires_grad = False

        self.num_latents = self.ae.num_latents
        self.latent_dim = self.ae.latent_dim
        
        if self.accelerator.is_main_process:
            # Use the same config hash for wandb run name as used for directory
            cfg_str = OmegaConf.to_yaml(cfg, resolve=True)
            cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()
            
            wandb_init_kwargs = {"dir": str(self.output_dir)}
            if cfg.general.wandb_name:
                wandb_init_kwargs["name"] = f"{cfg.general.wandb_name}-{cfg_hash[:8]}"
            else:
                wandb_init_kwargs["name"] = f"cfm_diffusion-{cfg_hash[:8]}"
            
            self.accelerator.init_trackers(
                project_name="ldlm_diffusion",
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_init_kwargs}
            )

        print("==> Creating ConditionalFlowMatcher...")
        self.fm = ConditionalFlowMatcher()

        cfg_dit = DiTConfig()
        cfg_dit.dim = cfg.model.dim
        cfg_dit.num_latents = self.num_latents
        cfg_dit.latent_dim = self.latent_dim
        cfg_dit.num_layers = cfg.model.num_layers
        cfg_dit.expansion_factor = cfg.model.expansion_factor
        cfg_dit.dev = self.accelerator.device

        self.v_predictor = DiTModel(cfg_dit)
        self.ema_decay = cfg.training.ema_decay
        self.ema_model = copy.deepcopy(self.v_predictor)

        self.opt = get_adamw_optimizer(
            self.v_predictor.parameters(),
            lr=cfg.training.optimizer.learning_rate,
            betas=cfg.training.optimizer.adam_betas,
            weight_decay=cfg.training.optimizer.adam_weight_decay
        )

        self.lr_scheduler = get_scheduler(
            cfg.training.optimizer.lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=cfg.training.optimizer.lr_warmup_steps * self.num_dev,
            num_training_steps=cfg.training.train_num_steps * self.num_dev,
        )

        # Check if we should use bin files or the old dataset loading
        self.use_bin_files = hasattr(cfg.data, 'train_bin_pattern') and cfg.data.train_bin_pattern is not None
        
        if self.use_bin_files:
            # Use new bin file data loading
            if self.accelerator.is_main_process:
                self.accelerator.print("Using .bin file data loading")
                self.accelerator.print(f"Train bin pattern: {cfg.data.train_bin_pattern}")
                if hasattr(cfg.data, 'val_bin_pattern') and cfg.data.val_bin_pattern:
                    self.accelerator.print(f"Val bin pattern: {cfg.data.val_bin_pattern}")
            
            # Get rank and world size from accelerator
            rank = self.accelerator.process_index
            world_size = self.accelerator.num_processes
            
            # Create a config that combines training settings with VAE model settings for bin files
            bin_cfg = OmegaConf.create({
                'training': {
                    'train_bs': cfg.training.train_bs,
                    'eval_bs': cfg.training.eval_bs
                },
                'model': {
                    'max_seq_len': loaded_vae_cfg.model.max_seq_len
                }
            })
            
            self.dataloader = get_dataloader_lvae_bin(
                bin_cfg, 
                cfg.data.train_bin_pattern, 
                rank, 
                world_size,
                tokenizer=self.tokenizer
            )
            
            self.val_dataloader = get_dataloader_lvae_bin(
                bin_cfg, 
                cfg.data.val_bin_pattern, 
                rank, 
                world_size,
                tokenizer=self.tokenizer
            )
            
            # For bin files, we don't have a fixed dataset size for validation samples
            self.num_samples_eval = 1000  # Arbitrary number for bin files
        else:
            # Use old dataset loading approach
            if self.accelerator.is_main_process:
                self.accelerator.print("Using traditional dataset loading")
            
            dataset = get_dataset(cfg.data.dataset_name) #, shard_size=cfg.data.shard_size) # FIXME make this flexibly optional
            self.dataset = dataset.shuffle(seed=cfg.general.seed)
            self.num_samples_eval = len(self.dataset['valid'])

            # Use the loaded VAE config for the dataloader
            loaded_vae_cfg.training.train_bs = cfg.training.train_bs
            self.dataloader = get_dataloader_lvae(
                loaded_vae_cfg, self.dataset['train'], self.tokenizer,
                loaded_vae_cfg.model.max_seq_len
            )
            
            loaded_vae_cfg.training.eval_bs = cfg.training.eval_bs
            self.val_dataloader = get_dataloader_lvae(
                loaded_vae_cfg, self.dataset['valid'], self.tokenizer,
                loaded_vae_cfg.model.max_seq_len, shuffle=False
            )

        self.train_num_steps = cfg.training.train_num_steps
        self.step = 0

        self.v_predictor, self.ema_model, self.opt, self.lr_scheduler, self.dataloader, self.val_dataloader = self.accelerator.prepare(
            self.v_predictor, self.ema_model, self.opt, self.lr_scheduler, self.dataloader, self.val_dataloader)

        # Handle checkpoint loading/resuming
        if cfg.general.get('checkpoint_path') is not None:
            if self.accelerator.is_main_process:
                print(f"Loading checkpoint from: {cfg.general.checkpoint_path}")
                print(f"Successfully Loaded CFM checkpoint!")
            self.load_from_checkpoint(cfg.general.checkpoint_path, resume_training=True)
        elif cfg.general.resume_training and cfg.general.resume_dir is not None:
            if self.accelerator.is_main_process:
                print(f"Resuming training from: {cfg.general.resume_dir}")
                print(f"Successfully Loaded CFM checkpoint!")
            checkpoint_file = Path(cfg.general.resume_dir) / 'model.pt'
            if checkpoint_file.exists():
                self.load_for_training(str(checkpoint_file), resume_training=True)

        model_size = sum(p.numel() for p in self.v_predictor.parameters())
        self.accelerator.print(f"Model params: {model_size / 1e6:.2f} M")

        # Create data iterators based on loading method
        if self.use_bin_files:
            # For bin files, check if we need to resume from a specific position
            if (cfg.general.get('checkpoint_path') is not None or 
                (cfg.general.resume_training and cfg.general.resume_dir is not None)) and self.step > 0:
                # We're resuming training, so wind the data generator to the correct position
                from dataset_util.dataset_helper import wind_data_generator_cfm
                
                if self.accelerator.is_main_process:
                    print(f"Winding CFM data generator to step {self.step}")
                
                # Get rank and world size
                rank = self.accelerator.process_index
                world_size = self.accelerator.num_processes
                
                # Create a config that combines training settings with VAE model settings for winding
                wind_cfg = OmegaConf.create({
                    'training': {
                        'train_bs': cfg.training.train_bs,
                        'gradient_accumulate_every': cfg.training.grad_accumulate
                    },
                    'model': {
                        'max_seq_len': loaded_vae_cfg.model.max_seq_len
                    }
                })
                
                # Wind the training data generator
                self.data_iter = wind_data_generator_cfm(
                    wind_cfg,
                    cfg.data.train_bin_pattern,
                    self.step,
                    rank,
                    world_size,
                    tokenizer = self.tokenizer
                )
                
                # For validation, we don't need to wind (always start from beginning)
                self.val_iter = get_dataloader_lvae_bin(
                    wind_cfg,
                    cfg.data.val_bin_pattern,
                    rank,
                    world_size,
                    tokenizer=self.tokenizer
                )
            else:
                # Normal startup, use the already created data loaders
                self.data_iter = self.dataloader
                self.val_iter = self.val_dataloader
        else:
            # Traditional data loaders need to be wrapped with cycle
            self.data_iter = cycle(self.dataloader)
            self.val_iter = cycle(self.val_dataloader)
            
        self.best_val_loss = float('inf')

    def save(self, file_name='model.pt') -> None:
        if not self.accelerator.is_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.v_predictor),
            'ema_model': self.accelerator.get_state_dict(self.ema_model),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
            'scheduler': self.lr_scheduler.state_dict(),
            'version': __version__,
            'cfg': OmegaConf.to_container(self.cfg, resolve=True),
            'best_val_loss': self.best_val_loss
        }

        model_save_path = self.output_dir / file_name
        torch.save(data, model_save_path)

        # We don't need to re-save the VAE, it's already in its own directory
        print(f'Saved diffusion model checkpoint to {model_save_path}')

    def load_for_generation(self, file_path) -> None:
        """Load model for generation/inference only."""
        data = torch.load(file_path, map_location="cpu", weights_only=False)

        model = self.accelerator.unwrap_model(self.v_predictor)
        ema_model = self.accelerator.unwrap_model(self.ema_model)
        model.load_state_dict(data['model'])
        
        if self.accelerator.is_local_main_process:
            ema_model.load_state_dict(data['ema_model'])

        model.to(self.accelerator.device)
        ema_model.to(self.accelerator.device) 

        self.step = data['step']
        if 'best_val_loss' in data:
            self.best_val_loss = data['best_val_loss']

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def load_for_training(self, file_path, resume_training=False) -> None:
        """Load model for training with full state restoration."""
        data = torch.load(file_path, map_location="cpu", weights_only=False)

        model = self.accelerator.unwrap_model(self.v_predictor)
        ema_model = self.accelerator.unwrap_model(self.ema_model)
        model.load_state_dict(data['model'])
        
        if 'ema_model' in data:
            ema_model.load_state_dict(data['ema_model'])

        model.to(self.accelerator.device)
        ema_model.to(self.accelerator.device) 

        # Load training state
        if 'step' in data:
            self.step = data['step']
            if self.accelerator.is_main_process:
                print(f"Resuming from step: {self.step}")

        if 'best_val_loss' in data:
            self.best_val_loss = data['best_val_loss']
            if self.accelerator.is_main_process:
                print(f"Best validation loss: {self.best_val_loss}")

        if resume_training:
            if 'opt' in data:
                self.opt.load_state_dict(data['opt'])
                if self.accelerator.is_main_process:
                    print("Loaded optimizer state.")
            
            if 'scheduler' in data:
                self.lr_scheduler.load_state_dict(data['scheduler'])
                if self.accelerator.is_main_process:
                    print("Loaded scheduler state.")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
            if self.accelerator.is_main_process:
                print("Loaded scaler state.")

        if self.accelerator.is_main_process:
            print("Checkpoint loading complete.")

    def load_from_checkpoint(self, checkpoint_path, resume_training=False) -> None:
        """Load model from external checkpoint directory."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
        
        # Try to load from model_best.pt first, then model.pt
        checkpoint_file = None
        for filename in ['model_best.pt', 'model.pt']:
            potential_path = checkpoint_path / filename
            if potential_path.exists():
                checkpoint_file = potential_path
                break
        
        if checkpoint_file is None:
            raise FileNotFoundError(f"No checkpoint file (model_best.pt or model.pt) found in {checkpoint_path}")
        
        if self.accelerator.is_main_process:
            print(f"Loading checkpoint from: {checkpoint_file}")
        
        self.load_for_training(str(checkpoint_file), resume_training=resume_training)

    # Keep the original load method for backward compatibility
    def load(self, file_path):
        """Legacy load method - redirects to load_for_generation."""
        return self.load_for_generation(file_path)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        last_val_loss = float('nan')
        last_ema_val_loss = float('nan')
    
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                
                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.data_iter)
                    batch = {k: v.to(device) for k, v in batch.items()}

                    with accelerator.accumulate(self.v_predictor):
                        with torch.no_grad():
                            latent = self.ae.get_latents(input_ids=batch['input_ids'], attn_mask=batch.get('attention_mask'))
                        
                        latent = latent.to(dtype=self.model_dtype)

                        x_0 = torch.randn_like(latent)
                        t, x_t, u_t = self.fm.get_sample_location_and_conditional_flow(x_0, latent)
                        v_t = self.v_predictor(x_t, t, x_0)  # we need to condition on the initial conditions (x_0) for flow matching (bc it's an ode solver) 
                        loss = F.mse_loss(v_t.float(), u_t.float())
                        
                        total_loss += loss.detach() / self.gradient_accumulate_every
                        accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    grad_norm = compute_grad_norm(self.v_predictor.parameters())
                    accelerator.clip_grad_norm_(self.v_predictor.parameters(), 1.0)

                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    
                    if (self.step + 1) % self.ema_update_every == 0:
                        ema(self.v_predictor, self.ema_model, self.ema_decay)

                    if accelerator.is_main_process:
                        # Calculate epoch differently for bin files vs traditional datasets
                        if self.use_bin_files:
                            # Calculate tokens processed: step * batch_size * grad_accumulate * num_devices * seq_len
                            # Note: we need to get max_seq_len from the loaded VAE config
                            max_seq_len = getattr(self.cfg.model, 'max_seq_len', 1024)  # fallback if not in config
                            tokens_processed = self.step * self.train_bs * self.gradient_accumulate_every * self.num_dev * max_seq_len
                            # Assume 100B tokens in dataset (can be made configurable)
                            total_dataset_tokens = getattr(self.cfg.data, 'total_tokens', 100_000_000_000)  # 100B default
                            epoch = tokens_processed / total_dataset_tokens
                            
                            # Log progress for bin files
                            if self.step % 500 == 0:  # Log every 500 steps
                                progress_pct = (tokens_processed / total_dataset_tokens) * 100
                                tokens_b = tokens_processed / 1_000_000_000  # Convert to billions
                                total_b = total_dataset_tokens / 1_000_000_000
                                accelerator.print(f"Progress: {progress_pct:.2f}% ({tokens_b:.2f}B / {total_b:.1f}B tokens)")
                        else:
                            epoch = (self.step * self.gradient_accumulate_every) / len(self.dataloader)
                            
                        logs = {
                            "loss": total_loss.item(),
                            "learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "grad_norm": grad_norm,
                            "step": self.step,
                            "epoch": epoch,
                            "samples": self.step * self.train_bs * self.gradient_accumulate_every * self.num_dev, 
                        }

                        if self.step > 0 and self.step % self.eval_every == 0:
                            self.v_predictor.eval()
                            self.ema_model.eval()

                            total_val_loss = 0.
                            total_ema_val_loss = 0.
                            
                            num_val_batches = 10
                            
                            for _ in range(num_val_batches):
                                val_batch = next(self.val_iter)
                                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                                with torch.no_grad():
                                    latent = self.ae.get_latents(input_ids=val_batch['input_ids'], attn_mask=val_batch.get('attention_mask'))
                                
                                latent = latent.to(dtype=self.model_dtype)
                                x_0 = torch.randn_like(latent)
                                t, x_t, u_t = self.fm.get_sample_location_and_conditional_flow(x_0, latent)
                                
                                v_t = self.v_predictor(x_t, t, x_0)
                                total_val_loss += F.mse_loss(v_t.float(), u_t.float()).item()

                                v_t = self.ema_model(x_t, t, x_0)
                                total_ema_val_loss += F.mse_loss(v_t.float(), u_t.float()).item()

                            last_val_loss = total_val_loss / num_val_batches
                            last_ema_val_loss = total_ema_val_loss / num_val_batches
                            
                            logs["val_loss"] = last_val_loss
                            logs["val_ema_loss"] = last_ema_val_loss
                            
                            self.save() # Save the latest model every evaluation
                            if last_ema_val_loss < self.best_val_loss:
                                self.accelerator.print(f"New best EMA validation loss: {last_ema_val_loss:.4f}. Saving model_best.pt")
                                self.best_val_loss = last_ema_val_loss
                                self.save(file_name='model_best.pt')
                                logs["val/best_ema_loss"] = self.best_val_loss
                            
                            self.v_predictor.train()
                        
                        accelerator.log(logs, step=self.step)
                        logs["val_loss"] = last_val_loss
                        logs["val_ema_loss"] = last_ema_val_loss
                        pbar.set_postfix(**logs)
                        
                    total_loss = 0.
                    self.step += 1
                    pbar.update(1)

                    if accelerator.is_main_process and self.step > 0 and (self.step % self.save_and_sample_every == 0):
                        self.eval()

    @torch.no_grad()
    def eval(self, verbose: bool = False) -> None:
        accelerator = self.accelerator
        model_for_generation = self.accelerator.unwrap_model(self.ema_model)
        model_for_generation.eval()
        self.ae.eval()

        generated_texts = []
        pbar = tqdm(range(0, self.num_gen_samples, self.gen_batch_size), desc=f"Generating {self.num_gen_samples} samples")

        for i in pbar:
            batch_size = min(self.gen_batch_size, self.num_gen_samples - i)
            
            latents = gen_samples(
                model=model_for_generation,
                num_latents=self.num_latents,
                dim_latents=self.latent_dim,
                batch_size=batch_size,
                accelerator=self.accelerator,
                steps=self.gen_steps,
                target_dtype=self.model_dtype,
                method="euler"
            )
            with torch.no_grad():
                # This seems to be a custom function in the old VAE, replacing with the standard one
                output_ids_list = self.ae.decode_latent(latents)
                output_ids = torch.argmax(output_ids_list, dim=-1)

            decoded_batch = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            generated_texts.extend(decoded_batch)

            if verbose: 
                for idx, decoded in enumerate(decoded_batch):
                    print(f"Sample {idx}: {decoded}")
                    print()
                    print()

            try:
                table = wandb.Table(columns=["step", "sample_id", "generated_text"])
                for j, text in enumerate(decoded_batch):
                    table.add_data(self.step, i+j, text)
                self.accelerator.log({"generated_samples": table}, step=self.step)
            except ImportError:
                print("wandb not installed, skipping logging of generated samples.")

        pbar.set_description(f"Generated {len(generated_texts)} samples")
        # print(generated_texts)
        # print("\nEvaluation finished.")
        # Make sure to set the model back to train mode
        model_for_generation.train()
