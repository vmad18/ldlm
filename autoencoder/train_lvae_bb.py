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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from einops import rearrange, reduce, repeat
from math import sqrt, log

from typing import Tuple

from transformers import BartForConditionalGeneration, get_scheduler, AutoTokenizer, PreTrainedTokenizerBase, \
    T5ForConditionalGeneration, AutoModelForCausalLM
from datasets import concatenate_datasets
from accelerate import Accelerator, DistributedDataParallelKwargs

from dataset_util.dataset_helper import get_dataset

from PIL import Image
from tqdm.auto import tqdm


import wandb

from datetime import datetime

import evaluation
from transformers.modeling_outputs import BaseModelOutput

generate_kwargs = {
    'beam': 
    {'max_length':64, 'min_length':5, 'do_sample':False, 'num_beams':4, 'no_repeat_ngram_size':0, 'repetition_penalty':1.2},
    'nucleus':
    {'max_length':64, 'min_length':5, 'do_sample':True, 'top_p':.95, 'num_beams':1, 'no_repeat_ngram_size':0, 'repetition_penalty':1.2}}


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


def get_output_dir(args):
    model_dir = f'{Path(args.dataset_name).stem}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    output_dir = os.path.join(args.save_dir, model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Created {output_dir}')
    return output_dir

class Trainer(object):

    def __init__(self,
                 args,
                 init_lr=1e-4,
                 lr_schedule="cosine",
                 adam_betas=(0.9, 0.99),
                 adam_weight_decay=0.01,
                 seed=412,
                 ) -> None:
        super().__init__()

        set_seeds(seed)

        self.args = args

        self.sample_every = args.sample_every
        self.num_samples_to_gen = args.num_samples_to_gen

        self.best_val_metric = 0
        self.num_samples = args.num_samples

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision = args.mixed_precision, 
            log_with="wandb",
            kwargs_handlers=[ddp_kwargs]
        )

        self.num_dev = self.accelerator.num_processes

        if self.accelerator.is_main_process:
            if args.output_dir is None:
                args.output_dir = get_output_dir(args)
            results_folder = args.output_dir

            config_dict = args.__dict__
            with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
                json.dump(config_dict, f, indent=2)
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(
                    run, config=config_dict,
                    init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}}
                )
            else:
                self.accelerator.init_trackers(
                    run, config=config_dict,
                    init_kwargs={"wandb": {"dir": results_folder}}
                )

        if args.freeze_bb == "freeze":
            print("Using Frozen T5/BART Backbone")
            ctx = torch.no_grad()
        else:
            ctx = nullcontext()


        if args.bb == "bart":
            from .bart_latent_model import LatentAEModel, get_latent_ae_tokenizer
            from dataset_util.dataset_helper import get_dataloader_lvae_t5 as get_dataloader # TODO CHANGE THIS BACK
            self.model, self.tokenizer, config = get_latent_ae_tokenizer(args, ctx, self.num_dev)
            self.model: LatentAEModel = self.model
            self.model = torch.compile(self.model, mode="max-autotune-no-cudagraphs")
        elif args.bb == "t5":
            from .t0_pp_lvae import LatentVAEModel, get_latent_vae_tokenizer_t5
            from dataset_util.dataset_helper import get_dataloader_lvae_t5 as get_dataloader
            self.model, self.tokenizer, config = get_latent_vae_tokenizer_t5(args, ctx, self.num_dev)
            self.model: LatentVAEModel = self.model
            # self.model = torch.compile(self.model, mode="max-autotune-no-cudagraphs")
        else:
            raise Exception("i dunno, erm")


        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.accelerator.is_main_process:
            self.accelerator.print(f'num trainable params: {num_trainable_params}')

        self.eval_every = args.eval_every

        self.train_bs = args.train_bs
        self.eval_bs = args.eval_bs

        self.num_train_steps = args.num_train_steps

        self.dataset = get_dataset(args.dataset_name)

        if args.eval:
            self.dataset["train"] = self.dataset["train"].select(range(1000))

        self.dataloader = get_dataloader(args, self.dataset["train"], config, self.tokenizer, args.max_seq_len,
                                          context_tokenizer=self.tokenizer,
                                          use_precomputed_latents=args.use_precomputed_latents,
                                          precomputed_latent_path=args.precomputed_latent_path,
                                          batch_size=args.train_bs)
        self.val_dataloader = get_dataloader(args, self.dataset['valid'], config, self.tokenizer, args.max_seq_len,
                                             shuffle=False, context_tokenizer=self.tokenizer,
                                             use_precomputed_latents=args.use_precomputed_latents,
                                             precomputed_latent_path=args.precomputed_latent_path,
                                             batch_size=args.eval_bs)
        self.max_seq_len = args.max_seq_len

        self.opt = get_adamw_optimizer(self.model.parameters(), lr=init_lr, betas=adam_betas,
                                       weight_decay=adam_weight_decay)

        self.grad_accumulate = args.grad_accumulate
        
        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=args.lr_warmup_steps * self.num_dev,
            num_training_steps=args.num_train_steps * self.num_dev,
        )

        if self.accelerator.is_main_process:
            self.save_dir = Path(args.save_dir)
            self.save_dir.mkdir(exist_ok=True)

        self.step = 0

        self.model, self.opt, self.dataloader, self.lr_scheduler, self.val_dataloader = self.accelerator.prepare(
            self.model, self.opt, self.dataloader, lr_scheduler, self.val_dataloader)
        
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)

        self.vae_loss_weight = 0.005
        self.kld_weight = args.kld_weight

        self.latent_dim = args.latent_dim
        self.num_latents = args.num_latents

    def save(self):
        if not self.accelerator.is_main_process:
            return

        model_path = os.path.join(self.args.output_dir, 'vae.pt')
        
        # Only save the VAE components, not the pretrained T5 backbone
        model = self.accelerator.unwrap_model(self.model)
        vae_state_dict = {
            'vae': model.vae.state_dict(),
            'latent_dim': model.latent_dim,
            'num_latents': model.num_latents,
            'use_precomputed_latents': model.use_precomputed_latents
        }
        
        data = {
            'args': self.args,
            'vae_state_dict': vae_state_dict
        }

        torch.save(data, model_path)
        print(f'Saved VAE components and args to {model_path}')

    @torch.no_grad()
    def sample_from_prior(self):
        """
        Samples from the VAE by decoding random latents from the prior (N(0,I)).
        """
        if not self.accelerator.is_main_process:
            return
        
        self.model.eval()

        model_to_generate_with = self.accelerator.unwrap_model(self.model)

        latents = torch.randn(
            (self.num_samples_to_gen, model_to_generate_with.num_latents, model_to_generate_with.latent_dim),
            device=self.accelerator.device
        ).to(model_to_generate_with.dtype)

        # Use the decode_latent method from our VAE model
        generated_ids = model_to_generate_with.decode_latent(latents, **generate_kwargs['nucleus'])
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        print("\n" + "─" * 50)
        print(f"Sampling from prior at step {self.step}")
        for i, text in enumerate(generated_texts):
            print(f"  Sample {i+1}: {text}")
        print("─" * 50 + "\n")

        # Log to wandb
        try:
            table = wandb.Table(columns=["step", "sample_id", "generated_text"])
            for i, text in enumerate(generated_texts):
                table.add_data(self.step, i + 1, text)
            self.accelerator.log({"generated_samples_from_prior": table}, step=self.step)
        except Exception as e:
            print(f"Wandb logging for samples failed: {e}")

        self.model.train()

    def load(self, file_path=None, resume_training=False):
        if file_path is None:
            file_path = self.args.resume_dir
        
        if not os.path.exists(file_path):
            print(f"Checkpoint not found at {file_path}. Starting from scratch.")
            return

        model_path = os.path.join(file_path, 'vae.pt')
        if not os.path.exists(model_path):
            print(f"vae.pt not found in {file_path}. Starting from scratch.")
            return

        data = torch.load(model_path, map_location='cpu')

        # Handle both new (VAE components only) and old (full model) checkpoint formats
        if 'vae_state_dict' in data:
            # New format: only VAE components
            vae_components = data['vae_state_dict']
            model = self.accelerator.unwrap_model(self.model)
            model.vae.load_state_dict(vae_components['vae'])
            print("Loaded VAE components from new checkpoint format.")
        elif 'model_state_dict' in data:
            # Old format: full model state dict
            model_state_dict = data['model_state_dict']
            self.model.load_state_dict(model_state_dict)
            print("Loaded full model from old checkpoint format.")
        else:
            # Very old format: the file is the state dict itself
            print("Loading VAE from an older checkpoint format.")
            self.model.load_state_dict(data)

        if resume_training:
            # If we need to resume optimizer state, etc., it would be loaded here.
            # For now, we only load model weights.
            print("Resuming training with loaded model weights.")
        else:
            print(f"Loaded model weights from {model_path}")

    def validation(self):
        self.model.eval()
        total_val_loss = 0.
        total_val_recon_loss = 0.
        total_val_kld_loss = 0.

        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                losses = self.model(**batch)
                if self.args.use_precomputed_latents:
                    recon_loss = losses['reconstruction_loss']
                    kld_loss = losses['kld_loss']
                    loss = recon_loss + self.kld_weight * kld_loss

                    total_val_recon_loss += recon_loss.item()
                    total_val_kld_loss += (self.kld_weight * kld_loss.item())
                else:
                    loss = losses['lm_loss'] + self.vae_loss_weight * losses['vae_loss']
                
                total_val_loss += loss.item()

                if self.accelerator.is_main_process and i == 0:
                    # use the encoder_outputs from the model's forward pass
                    enc_outs = losses['encoder_outputs']
                    if hasattr(self, 'use_ema') and self.use_ema:
                        generated_ids = self.ema.ema_model.generate(encoder_outputs=enc_outs, **generate_kwargs["beam"])
                    else:
                        generated_ids = self.model.generate(encoder_outputs=enc_outs, **generate_kwargs["beam"])

                    generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    for j in range(len(generated_text)):
                        self.accelerator.print(f"Sample {j}: {generated_text[j]}")

        num_batches = len(self.val_dataloader)
        avg_val_loss = total_val_loss / num_batches
        
        log_data = {"val/loss": avg_val_loss}
        if self.args.use_precomputed_latents:
            log_data["val/recon_loss"] = total_val_recon_loss / num_batches
            log_data["val/kld_loss"] = total_val_kld_loss / num_batches

        self.accelerator.log(log_data, step=self.step)
        self.model.train()

    def train(self):
        device = self.accelerator.device

        # --- Comprehensive Warm-up Pass ---
        if self.accelerator.is_main_process:
            self.accelerator.print("Starting comprehensive warm-up pass to compile graphs...")

        warmup_data = {k: v.to(device) for k, v in next(self.data_iter).items()}
        with self.accelerator.autocast():
            losses = self.model(**warmup_data)
            if self.args.use_precomputed_latents:
                loss = (losses['reconstruction_loss'] + self.kld_weight * losses['kld_loss']) / self.grad_accumulate
            else:
                loss = (losses['lm_loss'] + self.vae_loss_weight * losses['vae_loss']) / self.grad_accumulate

        self.accelerator.backward(loss)
        # self.opt.step()
        self.opt.zero_grad()
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self.accelerator.print("Warm-up pass complete.")
        # --- End Warm-up Pass ---

        with tqdm(initial=self.step, total=self.num_train_steps, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < self.num_train_steps:
                self.model.train()

                total_loss = 0.
                total_recon_loss = 0.
                total_kld_loss = 0.

                for i in range(self.grad_accumulate):
                    batch = {k: v.to(device) for k,v in next(self.data_iter).items()}

                    with self.accelerator.autocast():
                        # torch.compiler.cudagraph_mark_step_begin()
                        losses = self.model(**batch)

                        if self.args.use_precomputed_latents:
                            # Loss for VAE-only training on latents
                            recon_loss = losses['reconstruction_loss']
                            kld_loss = losses['kld_loss']
                            
                            total_recon_loss += recon_loss.item() / self.grad_accumulate
                            total_kld_loss += (kld_loss.item()) / self.grad_accumulate
                            loss = recon_loss + self.kld_weight * kld_loss
                        else:
                            # Loss for end-to-end training
                            loss = losses['lm_loss'] + self.vae_loss_weight * losses['vae_loss']
                    
                    final_loss = loss / self.grad_accumulate
                    self.accelerator.backward(final_loss)
                    total_loss += final_loss.item()

                # grad clipping
                if self.accelerator.sync_gradients:
                    grad_norm = compute_grad_norm(self.model.parameters())
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad()

                self.step += 1
                
                log_dict = {'loss': total_loss}
                if self.args.use_precomputed_latents:
                    log_dict['recon_loss'] = total_recon_loss
                    log_dict['kld_loss'] = total_kld_loss
                pbar.set_postfix(**log_dict)
                pbar.update(1)

                if self.accelerator.is_main_process:
                    if self.step % self.eval_every == 0:
                        self.validation()
                        self.save()
                    
                    if self.sample_every > 0 and self.step > 0 and self.step % self.sample_every == 0:
                        self.sample_from_prior()

                    log_data = {'train/loss': total_loss, 'lr': self.lr_scheduler.get_last_lr()[0]}
                    if self.args.use_precomputed_latents:
                        log_data['train/recon_loss'] = total_recon_loss
                        log_data['train/kld_loss'] = total_kld_loss
                    if self.accelerator.sync_gradients:
                        log_data['train/grad_norm'] = grad_norm
                    self.accelerator.log(log_data, step=self.step)

        self.accelerator.print('Training complete')
        self.save()
