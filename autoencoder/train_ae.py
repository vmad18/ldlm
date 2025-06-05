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

from dataset_util.dataset_helper import get_dataset, get_dataloader

from PIL import Image
from tqdm.auto import tqdm

from bart_latent_model import LatentAEModel, get_latent_ae_tokenizer

import wandb


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

        if args.freeze:
            ctx = torch.no_grad()
        else:
            ctx = nullcontext()

        self.model, self.tokenizer, config = get_latent_ae_tokenizer(ctx)

        self.model: LatentAEModel = self.model

        self.model.compile()

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

        self.data_loader = get_dataloader(args, self.dataset["train"], config, self.tokenizer, args.max_seq_len,
                                          context_tokenizer=self.tokenizer)
        self.val_dataloader = get_dataloader(args, self.dataset['valid'], config, self.tokenizer, args.max_seq_len,
                                             shuffle=False, context_tokenizer=self.tokenizer)
        self.max_seq_len = args.max_seq_len

        self.opt = get_adamw_optimizer(self.model.parameters(), lr=init_lr, betas=adam_betas,
                                       weight_decay=adam_weight_decay)

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

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                data = {k: v.to(device) for k, v in next(self.data_iter).items()}
                with accelerator.autocast():

                    enc_outs = self.model.bart_autoencode(data["input_ids"], attn_mask=data["attention_mask"],
                                                          num_dev=self.num_dev)

                    loss = self.model(labels=data['labels'], encoder_outputs=enc_outs).loss

                total_loss += loss.item()
                self.accelerator.backward(loss)

                accelerator.wait_for_everyone()

                grad_norm = compute_grad_norm(self.model.parameters())

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1

                # Log to WandB
                if self.step % 50 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        total_val_loss = 0.
                        total_lm_val_loss = 0.
                        data = {k: v.to(device) for k, v in next(self.val_iter).items()}

                        enc_outs = self.model.bart_autoencode(data["input_ids"], attn_mask=data["attention_mask"],
                                                              num_dev=self.num_dev)

                        loss = self.model(labels=data['labels'], encoder_outputs=enc_outs).loss
                        if self.args.lm_mode == 'freeze':
                            total_lm_val_loss += self.model(input_ids=data['input_ids'],
                                                            attention_mask=data['attention_mask'],
                                                            labels=data['labels']).loss.item()
                        total_val_loss += loss.item()

                        logs = {"train/loss": total_loss, "val/loss": total_val_loss, "grad_norm": grad_norm,
                                "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step,
                                "epoch": (self.step) / len(self.dataloader),
                                "samples": self.step * self.train_bs * self.num_dev}
                        if self.args.lm_mode == 'freeze':
                            logs["val/lm_loss"] = total_lm_val_loss
                        pbar.set_postfix(**logs)

                    self.model.train()
                else:
                    logs = {"train/loss": total_loss, "grad_norm": grad_norm, "lr": self.lr_scheduler.get_last_lr()[0],
                            "step": self.step, "epoch": (self.step) / len(self.dataloader),
                            "samples": self.step * self.train_bs * self.num_dev}

                if accelerator.is_main_process:
                    accelerator.log(logs, step=self.step)

                if self.step % self.eval_every == 0:
                    # self.validation()
                    accelerator.wait_for_everyone()
                    self.save()
                    self.model.train()

                pbar.update(1)
        # self.validation()
        self.save()

        accelerator.print('training complete')
