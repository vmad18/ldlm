import math
import copy
from pathlib import Path
import random
from functools import partial
from collections import namedtuple, Counter
from multiprocessing import cpu_count
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from contextlib import nullcontext
import json
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from einops import rearrange, reduce, repeat
from math import sqrt, log

from typing import Tuple

from transformers import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs

from dataset_util.dataset_helper import get_dataset, get_dataloader, get_dataloader_lvae, get_dataloader_lvae_bin, get_val_dataloader_lvae_bin

from PIL import Image
from tqdm.auto import tqdm

from .latent_vae import LatentVAEModel, get_latent_vae_tokenizer

import wandb

from datetime import datetime

import evaluation

import torch.profiler
from torch.profiler import profile, ProfilerActivity
from omegaconf import DictConfig, OmegaConf


generate_kwargs = {
    'beam': 
    {'max_length':2048, 'min_length':5, 'do_sample':False, 'num_beams':4, 'no_repeat_ngram_size':0, 'repetition_penalty':1.2},
    'nucleus':
    {'max_length':2048, 'min_length':5, 'do_sample':True, 'top_p':.95, 'num_beams':1, 'no_repeat_ngram_size':0, 'repetition_penalty':1.2}}


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


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
    # implementation adapted from https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), p=2) for p in parameters]), p=2).item()
    return total_norm


class Trainer(object):

    def __init__(self,
                 cfg: DictConfig,
                 output_dir: str
                 ) -> None:
        super().__init__()

        set_seeds(cfg.seed)

        self.cfg = cfg

        self.best_val_loss = float('inf')
        self.num_samples = cfg.data.num_samples

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision = cfg.training.mixed_precision, 
            log_with="wandb",
            kwargs_handlers=[ddp_kwargs]
        )

        self.num_dev = self.accelerator.num_processes

        # Handle checkpoint loading: if loading from external checkpoint, use its config for model creation
        if cfg.model.get('latent_model_path') is not None:
            checkpoint_path = Path(cfg.model.latent_model_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
            
            config_path = checkpoint_path / 'config.yaml'
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            if self.accelerator.is_main_process:
                print(f"Loading config from checkpoint: {config_path}")
            
            # Preserve the latent_model_path from command line before merging
            original_latent_model_path = cfg.model.latent_model_path
            
            # Load the checkpoint config and use its model config
            checkpoint_cfg = OmegaConf.load(config_path)
            # Use the model config from checkpoint, but keep other configs from current run
            self.cfg = OmegaConf.merge(cfg, {"model": checkpoint_cfg.model})
            
            # Restore the latent_model_path so we know we're resuming
            self.cfg.model.latent_model_path = original_latent_model_path
            
            if self.accelerator.is_main_process:
                print("Using model architecture from checkpoint config")

        if self.accelerator.is_main_process:
            # Create a unique directory for each run based on its config hash
            cfg_str = OmegaConf.to_yaml(self.cfg, resolve=True)
            cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()
            self.results_folder = Path(output_dir) / cfg_hash
            self.results_folder.mkdir(parents=True, exist_ok=True)
            # save the config to the results folder
            with open(self.results_folder / "config.yaml", "w") as f:
                f.write(OmegaConf.to_yaml(self.cfg, resolve=True))
            print(f"Results folder for this run: {self.results_folder}")

            run = os.path.split(__file__)[-1].split(".")[0]
            
            # Use part of the hash to create a unique wandb run name
            wandb_init_kwargs = {"dir": str(self.results_folder)}
            if self.cfg.wandb_name:
                wandb_init_kwargs["name"] = f"{self.cfg.wandb_name}-{cfg_hash[:8]}"
            else:
                wandb_init_kwargs["name"] = f"run-{cfg_hash[:8]}"

            self.accelerator.init_trackers(
                run,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                init_kwargs={"wandb": wandb_init_kwargs}
            )

        self.model, self.tokenizer = get_latent_vae_tokenizer(self.cfg.model)
        # self.model = torch.compile(self.model, mode="max-autotune-no-cudagraphs")
        self.model: LatentVAEModel = self.model
        
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.accelerator.is_main_process:
            self.accelerator.print(f'num trainable params: {num_trainable_params}')

        self.eval_every = self.cfg.training.eval_every

        self.train_bs = self.cfg.training.train_bs
        self.eval_bs = self.cfg.training.eval_bs

        self.train_num_steps = self.cfg.training.train_num_steps

        # Check if we should use bin files or the old dataset loading
        self.use_bin_files = hasattr(self.cfg.data, 'train_bin_pattern') and self.cfg.data.train_bin_pattern is not None
        
        if self.use_bin_files:
            # Use new bin file data loading
            if self.accelerator.is_main_process:
                self.accelerator.print("Using .bin file data loading")
                self.accelerator.print(f"Train bin pattern: {self.cfg.data.train_bin_pattern}")
                if hasattr(self.cfg.data, 'val_bin_pattern') and self.cfg.data.val_bin_pattern:
                    self.accelerator.print(f"Val bin pattern: {self.cfg.data.val_bin_pattern}")
            
            # Get rank and world size from accelerator
            rank = self.accelerator.process_index
            world_size = self.accelerator.num_processes
            
            self.dataloader = get_dataloader_lvae_bin(
                self.cfg, 
                self.cfg.data.train_bin_pattern, 
                rank, 
                world_size,
                tokenizer=self.tokenizer
            )
            
            self.val_dataloader = get_dataloader_lvae_bin(
                self.cfg, 
                self.cfg.data.val_bin_pattern, 
                rank, 
                world_size,
                tokenizer=self.tokenizer
            )
        else:
            # Use old dataset loading approach
            if self.accelerator.is_main_process:
                self.accelerator.print("Using traditional dataset loading")
            
            self.dataset = get_dataset(self.cfg.data.dataset_name, shard_size=self.cfg.data.num_samples)

            if self.cfg.eval:
                self.dataset["train"] = self.dataset["train"].select(range(1000))

            self.dataloader = get_dataloader_lvae(self.cfg, self.dataset["train"], self.tokenizer, self.cfg.model.max_seq_len,
                                              context_tokenizer=self.tokenizer)
            self.val_dataloader = get_dataloader_lvae(self.cfg, self.dataset['valid'], self.tokenizer, self.cfg.model.max_seq_len,
                                                 shuffle=False)
        
        self.max_seq_len = self.cfg.model.max_seq_len

        self.opt = get_adamw_optimizer(self.model.parameters(), 
                                       lr = self.cfg.training.optimizer.learning_rate, betas = self.cfg.training.optimizer.adam_betas,
                                       weight_decay = self.cfg.training.optimizer.adam_weight_decay)

        self.grad_accumulate = self.cfg.training.grad_accumulate
        
        lr_scheduler = get_scheduler(
            self.cfg.training.optimizer.lr_schedule,
            optimizer = self.opt,
            num_warmup_steps = self.cfg.training.optimizer.lr_warmup_steps * self.num_dev,
            num_training_steps = self.cfg.training.train_num_steps * self.num_dev,
        )

        if self.accelerator.is_main_process:
            pass

        self.step = 0

        self.model, self.opt, self.dataloader, self.lr_scheduler, self.val_dataloader = self.accelerator.prepare(
            self.model, self.opt, self.dataloader, lr_scheduler, self.val_dataloader)
        
        # Handle checkpoint loading/resuming
        if self.cfg.model.get('latent_model_path') is not None:
            if self.accelerator.is_main_process:
                print(f"Loading checkpoint from: {self.cfg.model.latent_model_path}")
            self.load_from_checkpoint(self.cfg.model.latent_model_path, resume_training=True)
        elif self.cfg.resume_training and self.cfg.resume_dir is not None:
            if self.accelerator.is_main_process:
                print(f"Resuming training from: {self.cfg.resume_dir}")
            self.load(file_path=self.cfg.resume_dir, resume_training=True)
        
        # Create data iterators based on loading method - AFTER checkpoint loading so step is set
        if self.use_bin_files:
            # For bin files, check if we need to resume from a specific position
            if (self.cfg.model.get('latent_model_path') is not None or 
                (self.cfg.resume_training and self.cfg.resume_dir is not None)) and self.step > 0:
                # We're resuming training, so wind the data generator to the correct position
                from dataset_util.dataset_helper import wind_data_generator
                
                if self.accelerator.is_main_process:
                    print(f"Winding data generator to step {self.step}")
                
                # Get rank and world size
                rank = self.accelerator.process_index
                world_size = self.accelerator.num_processes
                
                # Wind the training data generator
                self.data_iter = wind_data_generator(
                    self.cfg,
                    self.cfg.data.train_bin_pattern,
                    self.step,
                    rank,
                    world_size,
                    tokenizer=self.tokenizer
                )
                
                # For validation, we don't need to wind (always start from beginning)
                self.val_iter = get_dataloader_lvae_bin(
                    self.cfg,
                    self.cfg.data.val_bin_pattern,
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

        self.kld_weight = self.cfg.training.kld_weight
        # KLD annealing: start at 0, linearly increase to target weight over first 2000 steps
        self.kld_annealing_steps = 2000

    def get_annealed_kld_weight(self):
        """Compute the current KLD weight with linear annealing."""
        if self.step < self.kld_annealing_steps:
            return self.kld_weight * (self.step / self.kld_annealing_steps)
        else:
            return self.kld_weight

    def save(self, file_name='model.pt'):
        if not self.accelerator.is_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
            'best_val_loss': self.best_val_loss
        }

        torch.save(data, str(self.results_folder / file_name))

    def load(self, file_path=None, resume_training=False):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(file_path / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if 'best_val_loss' in data:
            self.best_val_loss = data['best_val_loss']
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        if resume_training:
            for _ in range(self.step):
                self.lr_scheduler.step()

    def load_from_checkpoint(self, checkpoint_path, resume_training=False):
        """Load model from external checkpoint directory."""
        checkpoint_path = Path(checkpoint_path)
        accelerator = self.accelerator
        device = accelerator.device
        
        # Try to load from model_best.pt first, then model.pt
        checkpoint_file = None
        for filename in ['model_best.pt', 'model.pt']:
            potential_path = checkpoint_path / filename
            if potential_path.exists():
                checkpoint_file = potential_path
                break
        
        if checkpoint_file is None:
            raise FileNotFoundError(f"No checkpoint file (model_best.pt or model.pt) found in {checkpoint_path}")
        
        if accelerator.is_main_process:
            print(f"Loading checkpoint from: {checkpoint_file}")
        
        # Load checkpoint data
        data = torch.load(str(checkpoint_file), map_location=device, weights_only=False)
        
        # Load model state
        model = self.accelerator.unwrap_model(self.model)
        if 'model' in data:
            model.load_state_dict(data['model'])
            if accelerator.is_main_process:
                print("Successfully loaded model weights from 'model' key.")
        else:
            # Handle case where checkpoint is just the state_dict
            model.load_state_dict(data)
            if accelerator.is_main_process:
                print("Loaded model weights from raw state_dict.")
        
        # Load training state
        if 'step' in data:
            self.step = data['step']
            if accelerator.is_main_process:
                print(f"Resuming from step: {self.step}")
        
        if 'opt' in data:
            self.opt.load_state_dict(data['opt'])
            if accelerator.is_main_process:
                print("Loaded optimizer state.")
        
        if 'best_val_loss' in data:
            self.best_val_loss = data['best_val_loss']
            if accelerator.is_main_process:
                print(f"Best validation loss: {self.best_val_loss}")
        
        if exists(self.accelerator.scaler) and 'scaler' in data and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
            if accelerator.is_main_process:
                print("Loaded scaler state.")
        
        # Advance learning rate scheduler to current step
        if resume_training and hasattr(self, 'lr_scheduler'):
            if accelerator.is_main_process:
                print(f"Advancing learning rate scheduler by {self.step} steps...")
            for _ in range(self.step):
                self.lr_scheduler.step()
        
        if accelerator.is_main_process:
            print("Checkpoint loading complete.")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        accelerator = self.accelerator
        device = accelerator.device
        model = self.accelerator.unwrap_model(self.model)
        
        num_val_batches = 25
        total_val_loss = 0.
        total_recon_loss = 0.
        total_kld_loss = 0.

        for _ in range(num_val_batches):
            val_data = {k: v.to(device) for k, v in next(self.val_iter).items()}
            losses = model.autoencode(val_data["input_ids"], attn_mask=val_data["attention_mask"])
            
            val_loss = losses['reconstruction_loss'] + self.get_annealed_kld_weight() * losses['kld_loss']
            total_val_loss += val_loss.item()
            total_recon_loss += losses['reconstruction_loss'].item()
            total_kld_loss += losses['kld_loss'].item()

        avg_val_loss = total_val_loss / num_val_batches
        avg_recon_loss = total_recon_loss / num_val_batches
        avg_kld_loss = total_kld_loss / num_val_batches

        # Calculate epoch differently for bin files vs traditional datasets
        if self.use_bin_files:
            # Calculate tokens processed: step * batch_size * grad_accumulate * num_devices * seq_len
            tokens_processed = self.step * self.train_bs * self.grad_accumulate * self.num_dev * self.max_seq_len
            # Assume 100B tokens in dataset (can be made configurable)
            total_dataset_tokens = getattr(self.cfg.data, 'total_tokens', 100_000_000_000)  # 100B default
            epoch = tokens_processed / total_dataset_tokens
        else:
            epoch = (self.step * self.grad_accumulate) / len(self.dataloader)
            
        logs = {
            "val/loss": avg_val_loss,
            "val/reconstruction_loss": avg_recon_loss,
            "val/kld_loss": avg_kld_loss,
            "step": self.step,
            "epoch": epoch
        }
        
        self.save()
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save(file_name='model_best.pt')
            logs['val/best_loss'] = self.best_val_loss

        val_data = {k: v.to(device) for k, v in next(self.val_iter).items()}
        input_ids = val_data["input_ids"][:self.cfg.training.eval_bs]
        
        embeddings = model.embed(input_ids)
        attn_mask = val_data.get("attention_mask", torch.ones_like(input_ids))[:self.cfg.training.eval_bs]
        
        recon_embeds, _, _ = model.vae(embeddings, attn_mask.bool())
        recon_logits = model.dembed_head(recon_embeds[..., :input_ids.shape[1], :])
        recon_ids = torch.argmax(recon_logits, dim=-1)

        original_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        reconstructed_texts = self.tokenizer.batch_decode(recon_ids, skip_special_tokens=True)

        recon_table = wandb.Table(columns=["Original", "Reconstructed"])
        for orig, recon in zip(original_texts, reconstructed_texts):
            recon_table.add_data(orig, recon)
        logs["reconstructions"] = recon_table

        num_gen_samples = 4
        latents = torch.randn((num_gen_samples, model.num_latents, model.latent_dim), device=device)
        gen_logits = model.decode_latent(latents)
        gen_ids = torch.argmax(gen_logits, dim=-1)
        generated_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        gen_table = wandb.Table(columns=["Generated Text"])
        for gen in generated_texts:
            gen_table.add_data(gen)
        logs["generated_samples"] = gen_table

        accelerator.log(logs, step=self.step)
        self.model.train()

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        self.model.train()

        # --- Comprehensive Warm-up Pass ---
        if accelerator.is_main_process:
            accelerator.print("Starting comprehensive warm-up pass to compile forward and backward graphs...")

        # Get one batch for warm-up
        warmup_data = {k: v.to(device) for k, v in next(self.data_iter).items()}
        
        # Full forward, backward, and optimizer step to trigger compilation
        with accelerator.autocast():
            losses = self.model(warmup_data["input_ids"], attn_mask=warmup_data["attention_mask"])
            # Use the same loss calculation as in the main loop
            loss = (losses['reconstruction_loss'] + self.get_annealed_kld_weight() * losses['kld_loss']) / self.grad_accumulate

        self.accelerator.backward(loss)
        self.opt.step()
        self.opt.zero_grad() # Reset gradients immediately after warm-up
        
        accelerator.wait_for_everyone() # Ensure all processes are synchronized
        
        if accelerator.is_main_process:
            accelerator.print("Warm-up pass complete.")
        # --- End Warm-up Pass ---

        trace_handler = None
        if accelerator.is_main_process:
            profile_dir = str(self.results_folder / "torch_profile")
            trace_handler = torch.profiler.tensorboard_trace_handler(profile_dir)

        # with profile(
        #     activities=[ProfilerActivity.CUDA],
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=3),
        #     on_trace_ready=trace_handler,
        #     record_shapes=True,
        #     with_stack=True,
        #     experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
        # ) as prof:
        if True:
            with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
                while self.step < self.train_num_steps:
                    self.opt.zero_grad()
                    total_loss = 0.
                    total_recon_loss = 0.
                    total_kld_loss = 0.
                    for _ in range(self.grad_accumulate):
                        
                        data = {k: v.to(device) for k, v in next(self.data_iter).items()}
                        with accelerator.autocast():
                            torch.compiler.cudagraph_mark_step_begin()
                            losses = self.model(data["input_ids"], attn_mask=data["attention_mask"])
                            recon_loss = losses['reconstruction_loss']
                            kld_loss = losses['kld_loss']
                            loss = (recon_loss + self.get_annealed_kld_weight() * kld_loss) / self.grad_accumulate
                        
                        total_loss += loss.item()
                        total_recon_loss += recon_loss.item() / self.grad_accumulate
                        total_kld_loss += kld_loss.item() / self.grad_accumulate
                        self.accelerator.backward(loss)
                    
                    accelerator.wait_for_everyone()
                    grad_norm = compute_grad_norm(self.model.parameters())
                    accelerator.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.opt.step()
                    self.lr_scheduler.step()
                    accelerator.wait_for_everyone()

                    self.step += 1
                    log_dict = {'loss': total_loss}
                    log_dict['recon_loss'] = total_recon_loss
                    log_dict['kld_loss'] = total_kld_loss
                    pbar.set_postfix(**log_dict)
                    pbar.update(1)

                    if accelerator.is_main_process:
                        if self.step % 50 == 0:
                            current_kld_weight = self.get_annealed_kld_weight()
                            
                            # Calculate epoch differently for bin files vs traditional datasets
                            if self.use_bin_files:
                                # Calculate tokens processed: step * batch_size * grad_accumulate * num_devices * seq_len
                                tokens_processed = self.step * self.train_bs * self.grad_accumulate * self.num_dev * self.max_seq_len
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
                                epoch = (self.step * self.grad_accumulate) / len(self.dataloader)
                                
                            logs = {"train/loss": total_loss, 
                                    "train/recon_loss": total_recon_loss,
                                    "train/kld_loss": total_kld_loss,
                                    "train/kld_weight": current_kld_weight,
                                    "grad_norm": grad_norm,
                                    "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step,
                                    "epoch": epoch,
                                    "samples": self.step * self.train_bs * self.num_dev * self.grad_accumulate
                                }
                            accelerator.log(logs, step=self.step)
                        
                        
                        if self.step % self.eval_every == 0:
                            self.evaluate()

                    # prof.step()

        self.save()
        accelerator.print('training complete')
        return self.best_val_loss
