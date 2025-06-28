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
                 dataset_name,
                 train_bs=32,
                 eval_bs=64,
                 init_lr=1e-4,
                 train_num_steps=100000,
                 lr_schedule="cosine",
                 num_warmup_steps=500,
                 adam_betas=(0.9, 0.99),
                 adam_weight_decay=0.01,
                 num_samples=None,
                 eval_every=1000,
                 results_folder="./results",
                 mixed_percision="no",
                 grad_accumulate: int = 16,
                 seed=412,
                 ) -> None:
        super().__init__()

        set_seeds(seed)

        self.args = args

        self.best_val_metric = 0
        self.num_samples = num_samples

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            # mixed_percision = mixed_percision, 
            log_with="wandb",
            kwargs_handlers=[ddp_kwargs]
        )

        self.num_dev = self.accelerator.num_processes

        if self.accelerator.is_main_process:
            if args.output_dir is None:
                args.output_dir = get_output_dir(args)
            results_folder = args.output_dir
            with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args,
                                               init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})

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
            self.model = torch.compile(self.model, mode="max-autotune-no-cudagraphs")
        else:
            raise Exception("i dunno, erm")


        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.accelerator.is_main_process:
            self.accelerator.print(f'num trainable params: {num_trainable_params}')

        self.eval_every = eval_every

        self.train_bs = train_bs
        self.eval_bs = eval_bs

        self.train_num_steps = train_num_steps

        self.dataset = get_dataset(dataset_name)

        if args.eval:
            self.dataset["train"] = self.dataset["train"].select(range(1000))

        self.dataloader = get_dataloader(args, self.dataset["train"], config, self.tokenizer, args.max_seq_len,
                                          context_tokenizer=self.tokenizer,
                                          use_precomputed_latents=args.use_precomputed_latents,
                                          precomputed_latent_path=args.precomputed_latent_path,
                                          batch_size=train_bs)
        self.val_dataloader = get_dataloader(args, self.dataset['valid'], config, self.tokenizer, args.max_seq_len,
                                             shuffle=False, context_tokenizer=self.tokenizer,
                                             use_precomputed_latents=args.use_precomputed_latents,
                                             precomputed_latent_path=args.precomputed_latent_path,
                                             batch_size=eval_bs)
        self.max_seq_len = args.max_seq_len

        self.opt = get_adamw_optimizer(self.model.parameters(), lr=init_lr, betas=adam_betas,
                                       weight_decay=adam_weight_decay)

        self.grad_accumulate = grad_accumulate
        
        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps * self.num_dev,
            num_training_steps=train_num_steps * self.num_dev,
        )

        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        self.step = 0

        self.model, self.opt, self.dataloader, self.lr_scheduler, self.val_dataloader = self.accelerator.prepare(
            self.model, self.opt, self.dataloader, lr_scheduler, self.val_dataloader)
        
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)

        self.vae_loss_weight = 0.005
        self.kld_weight = 1e-6

        self.latent_dim = 1024
        self.num_latents = 32 

    def save(self):
        if not self.accelerator.is_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None
        }

        torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, file_path=None, resume_training=False):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(file_path / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        if resume_training:
            for _ in range(self.step):
                self.lr_scheduler.step()


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

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.
                total_recon_loss = 0.
                total_kld_loss = 0.

                for i in range(self.grad_accumulate):
                    batch = {k: v.to(device) for k,v in next(self.data_iter).items()}
                    
                    if self.accelerator.is_main_process and self.step == 0 and i == 0:
                        # print("\n--- DEBUG: First training batch ---", flush=True)
                        input_latents_tensor = batch['input_latents']
                        # print(f"DEBUG Trainer: batch['input_latents'].shape: {input_latents_tensor.shape}", flush=True)
                        # print(f"DEBUG Trainer: batch['input_latents'].dtype: {input_latents_tensor.dtype}", flush=True)
                        
                        if torch.isnan(input_latents_tensor).any():
                            print("DEBUG Trainer: !!! NaN FOUND IN BATCH TENSOR !!!", flush=True)
                        # else:
                            # print("DEBUG Trainer: Batch tensor is clean.", flush=True)

                        mean_val = input_latents_tensor.mean().item()
                        std_val = input_latents_tensor.std().item()
                        min_val = input_latents_tensor.min().item()
                        max_val = input_latents_tensor.max().item()

                        # print(f"DEBUG Trainer: batch['input_latents'].mean(): {mean_val}", flush=True)
                        # print(f"DEBUG Trainer: batch['input_latents'].std(): {std_val}", flush=True)
                        # print(f"DEBUG Trainer: batch['input_latents'].min(): {min_val}", flush=True)
                        # print(f"DEBUG Trainer: batch['input_latents'].max(): {max_val}", flush=True)

                        if std_val == 0:
                            print("DEBUG Trainer: !!! BATCH TENSOR HAS ZERO VARIANCE !!!", flush=True)

                    with self.accelerator.autocast():
                        # torch.compiler.cudagraph_mark_step_begin()
                        losses = self.model(**batch)

                        if self.args.use_precomputed_latents:
                            # Loss for VAE-only training on latents
                            recon_loss = losses['reconstruction_loss']
                            kld_loss = losses['kld_loss']
                            
                            total_recon_loss += recon_loss.item() / self.grad_accumulate
                            total_kld_loss += (self.kld_weight * kld_loss.item()) / self.grad_accumulate
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
                    
                    log_data = {'train/loss': total_loss, 'lr': self.lr_scheduler.get_last_lr()[0]}
                    if self.args.use_precomputed_latents:
                        log_data['train/recon_loss'] = total_recon_loss
                        log_data['train/kld_loss'] = total_kld_loss
                    if self.accelerator.sync_gradients:
                        log_data['train/grad_norm'] = grad_norm
                    self.accelerator.log(log_data, step=self.step)
                    
                    if self.step % 1000 == 0:
                        self.save()

        self.accelerator.print('Training complete')
