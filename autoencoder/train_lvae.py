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

from transformers import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs

from dataset_util.dataset_helper import get_dataset, get_dataloader, get_dataloader_lvae

from PIL import Image
from tqdm.auto import tqdm

from .latent_vae import LatentVAEModel, get_latent_vae_tokenizer

import wandb

from datetime import datetime

import evaluation

import torch.profiler
from torch.profiler import profile, ProfilerActivity


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
                 mixed_precision="bf16",
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
            mixed_precision = mixed_precision, 
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


        self.model, self.tokenizer = get_latent_vae_tokenizer(args)
        self.model = torch.compile(self.model, mode="max-autotune-no-cudagraphs")
        self.model: LatentVAEModel = self.model
        
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

        self.dataloader = get_dataloader_lvae(args, self.dataset["train"], self.tokenizer, args.max_seq_len,
                                          context_tokenizer=self.tokenizer)
        self.val_dataloader = get_dataloader_lvae(args, self.dataset['valid'], self.tokenizer, args.max_seq_len,
                                             shuffle=False)
        
        self.max_seq_len = args.max_seq_len

        self.opt = get_adamw_optimizer(self.model.parameters(), 
                                       lr = init_lr, betas = adam_betas,
                                       weight_decay = adam_weight_decay)

        self.grad_accumulate = grad_accumulate
        
        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer = self.opt,
            num_warmup_steps = num_warmup_steps * self.num_dev,
            num_training_steps = train_num_steps * self.num_dev,
        )

        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        self.step = 0

        self.model, self.opt, self.dataloader, self.lr_scheduler, self.val_dataloader = self.accelerator.prepare(
            self.model, self.opt, self.dataloader, lr_scheduler, self.val_dataloader)
        
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)

        self.kld_weight = 1e-6

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
            loss = (losses['reconstruction_loss'] + self.kld_weight * losses['kld_loss']) / self.grad_accumulate

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
                    for _ in range(self.grad_accumulate):
                        
                        data = {k: v.to(device) for k, v in next(self.data_iter).items()}
                        with accelerator.autocast():
                            torch.compiler.cudagraph_mark_step_begin()
                            losses = self.model(data["input_ids"], attn_mask=data["attention_mask"])
                            recon_loss = losses['reconstruction_loss']
                            kld_loss = self.kld_weight * losses['kld_loss']
                            loss = (recon_loss + kld_loss) / self.grad_accumulate
                        total_loss += loss.item()
                        self.accelerator.backward(loss)
                    
                    accelerator.wait_for_everyone()
                    grad_norm = compute_grad_norm(self.model.parameters())
                    accelerator.clip_grad_norm_(self.model.parameters(), 5.0)
                    self.opt.step()
                    self.lr_scheduler.step()
                    accelerator.wait_for_everyone()

                    if self.step % 50 == 0 and accelerator.is_main_process:
                        self.model.eval()
                        with torch.no_grad():
                            total_val_loss = 0.
                            val_data = {k: v.to(device) for k, v in next(self.val_iter).items()}
                            losses = self.model(val_data["input_ids"], attn_mask=val_data["attention_mask"])
                            val_loss = losses['reconstruction_loss'] + self.kld_weight * losses['kld_loss']
                            total_val_loss = val_loss.item()
                            logs = {"train/loss": total_loss, "val/loss": total_val_loss, "grad_norm": grad_norm,
                                    "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step,
                                    "epoch": (self.step * self.grad_accumulate) / len(self.dataloader),
                                    "samples": self.step * self.train_bs * self.num_dev * self.grad_accumulate
                                   }
                            pbar.set_postfix(**logs)
                            accelerator.log(logs, step=self.step)
                        self.model.train()
                    else:
                        logs = {"train/loss": total_loss, 
                                "grad_norm": grad_norm, 
                                "lr": self.lr_scheduler.get_last_lr()[0]
                               }
                        if accelerator.is_main_process:
                             accelerator.log(logs, step=self.step)

                    self.step += 1
                    pbar.update(1)
                    # prof.step()

        self.save()
        accelerator.print('training complete')
