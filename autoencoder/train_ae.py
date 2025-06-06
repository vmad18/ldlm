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

from datetime import datetime

import evaluation


generate_kwargs = {
    'beam': 
    {'max_length':64, 'min_length':5, 'do_sample':False, 'num_beams':4, 'no_repeat_ngram_size':3, 'repetition_penalty':1.2},
    'nucleus':
    {'max_length':64, 'min_length':5, 'do_sample':True, 'top_p':.95, 'num_beams':1, 'no_repeat_ngram_size':3, 'repetition_penalty':1.2}}


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
                 grad_accumulate: int = 4,
                 seed=412,
                 ) -> None:
        super().__init__()

        set_seeds(seed)

        self.args = args

        self.best_val_metric = 0
        self.num_samples = num_samples

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_percision = mixed_percision, 
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
            print("Freezing BART Backbone")
            ctx = torch.no_grad()
        else:
            ctx = nullcontext()

        self.model, self.tokenizer, config = get_latent_ae_tokenizer(args, ctx, self.num_dev)
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

        self.dataloader = get_dataloader(args, self.dataset["train"], config, self.tokenizer, args.max_seq_len,
                                          context_tokenizer=self.tokenizer)
        self.val_dataloader = get_dataloader(args, self.dataset['valid'], config, self.tokenizer, args.max_seq_len,
                                             shuffle=False, context_tokenizer=self.tokenizer)
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
        pred_text = {k:[] for k,_ in generate_kwargs.items()}    
        bart_text = {k:[] for k,_ in generate_kwargs.items()}    
        ref_text = []
        accelerator = self.accelerator
        device = self.accelerator.device
        for batch in tqdm(self.val_dataloader):
            for strategy in generate_kwargs.keys():
                gen_kwargs = generate_kwargs[strategy]
                gen_kwargs['max_length'] = self.max_seq_len
                data = {k:v.to(device) for k,v in batch.items()}
                # Compute generated language

                enc_outs = self.model.bart_autoencode(input_ids = data["input_ids"], attn_mask = data["attention_mask"])

                if self.num_dev > 1:
                    sample_ids = self.model.module.generate(encoder_outputs=enc_outs, **gen_kwargs)
                else:
                    sample_ids = self.model.generate(encoder_outputs=enc_outs, **gen_kwargs)
                
                # Pad sample_ids to max_seq_len
                sample_ids = F.pad(sample_ids, (0, self.max_seq_len - sample_ids.shape[-1]), value=self.tokenizer.pad_token_id)
                gathered_sample_ids = accelerator.gather(sample_ids).to('cpu')
                texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in gathered_sample_ids]
                pred_text[strategy].extend(texts_list)

                # Compute BART language
                if self.num_dev > 1:
                    sample_ids2 = self.model.module.generate(input_ids = data['input_ids'], attention_mask = data['attention_mask'], **gen_kwargs)
                else:
                    sample_ids2 = self.model.generate(input_ids = data['input_ids'], attention_mask = data['attention_mask'], **gen_kwargs)
                sample_ids2 = F.pad(sample_ids2, (0, self.max_seq_len - sample_ids2.shape[-1]), value=self.tokenizer.pad_token_id)
                gathered_sample_ids2 = accelerator.gather(sample_ids2).to('cpu')
                texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in gathered_sample_ids2]
                bart_text[strategy].extend(texts_list)

            # Store reference language
            gathered_input_ids = accelerator.gather(data['input_ids']).to('cpu')
            texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in gathered_input_ids]
            ref_text.extend(texts_list)
            if len(ref_text) > 1000:
                break

        if not self.accelerator.is_main_process:
            return
        # Compute metrics
        metrics = {}
        for strategy in generate_kwargs.keys():
            # Compute BLEU score
            metrics[f'autoencoder/{strategy}/bleu'] = evaluation.compute_bleu(pred_text[strategy], ref_text)
            metrics[f'bart/{strategy}/bleu'] = evaluation.compute_bleu(bart_text[strategy], ref_text)
            # Compute perplexity

            if all(pred_text[strategy]):
                metrics[f'autoencoder/{strategy}/perplexity'] = evaluation.compute_perplexity(pred_text[strategy])

            if all(bart_text[strategy]):
                metrics[f'bart/{strategy}/perplexity'] = evaluation.compute_perplexity(bart_text[strategy])

            rouge_metrics = evaluation.compute_rouge(pred_text[strategy], ref_text)
            for k,v in rouge_metrics.items():
                metrics[f'autoencoder/{strategy}/{k}'] = v
            rouge_metrics = evaluation.compute_rouge(bart_text[strategy], ref_text)
            for k,v in rouge_metrics.items():
                metrics[f'bart/{strategy}/{k}'] = v
        metrics['reference/perplexity'] = evaluation.compute_perplexity(ref_text)
         

        accelerator.log(metrics, self.step)

        # Log samples
        # reference | strategy0/autoencoder | strategy0/bart | strategy1/autoencoder | strategy1/bart | ...
        columns = ['reference'] + [f'{strategy}/autoencoder' for strategy in generate_kwargs.keys()] + [f'{strategy}/bart' for strategy in generate_kwargs.keys()]
        data = []
        for i in range(len(ref_text)):
            row = [ref_text[i]]
            for strategy in generate_kwargs.keys():
                row.append(pred_text[strategy][i])
            
            for strategy in generate_kwargs.keys():
                row.append(bart_text[strategy][i])
            data.append(row)
        table = wandb.Table(columns=columns, data=data)
        accelerator.log({f"Samples": table}, self.step)

    
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        self.model.train()

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                # for i in range(self.grad_accumulate):
                data = {k: v.to(device) for k, v in next(self.data_iter).items()}
                with accelerator.autocast():
                    enc_outs = self.model.bart_autoencode(data["input_ids"], attn_mask=data["attention_mask"])
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

                        enc_outs = self.model.bart_autoencode(data["input_ids"], attn_mask=data["attention_mask"])

                        loss = self.model(labels=data['labels'], encoder_outputs=enc_outs).loss
                        if self.args.freeze_bb == 'freeze':
                            total_lm_val_loss += self.model(input_ids=data['input_ids'],
                                                            attention_mask=data['attention_mask'],
                                                            labels=data['labels']).loss.item()
                        total_val_loss += loss.item()

                        logs = {"train/loss": total_loss, "val/loss": total_val_loss, "grad_norm": grad_norm,
                                "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step,
                                "epoch": (self.step) / len(self.dataloader),
                                "samples": self.step * self.train_bs * self.num_dev}
                        if self.args.freeze_bb == 'freeze':
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
                    self.validation()
                    accelerator.wait_for_everyone()
                    self.save()
                    self.model.train()

                pbar.update(1)
        self.validation()
        self.save()

        accelerator.print('training complete')
