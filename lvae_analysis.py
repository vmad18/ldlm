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

import wandb

from datetime import datetime

import evaluation

import torch.profiler
from torch.profiler import profile, ProfilerActivity
from omegaconf import DictConfig, OmegaConf

from autoencoder.train_lvae import Trainer 

import random 

import os 
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import sys


"""
this can def. be better, but it was quickest to impl. 
but, if it works it works :p
"""


class VAETest(object):

    def __init__(
                self, 
                cfg: DictConfig,
                output_dir: str = "./results_lvae_analysis") -> None:
        self.train = Trainer(cfg, output_dir)

    @torch.no_grad()
    def test_interp(self) -> None:
        self.train.model.eval()
        accelerator = self.train.accelerator
        device = self.train.accelerator.device
        model = self.train.accelerator.unwrap_model(self.train.model)
        
        num_val_batches = 25
        total_val_loss = 0.
        total_recon_loss = 0.
        total_kld_loss = 0.

        encoded_latents = []
        original_ids = [] 

        for _ in range(num_val_batches):
            val_data = {k: v.to(device) for k, v in next(self.train.val_iter).items()}
            latents = model.get_latents(val_data["input_ids"], attn_mask=val_data["attention_mask"], mu_only = True)
            original_ids.append(val_data["input_ids"])
            encoded_latents.append(latents)
        

        random_pairs = []
        for _ in range(40):
            pair = (random.randint(0, 24), 
                random.randint(0, 24))
            random_pairs.append(pair)

        print("==> Random pairs <==")
        print(random_pairs)
        for (latent_idx_1, latent_idx_2) in random_pairs:
            for alpha in [0.1, 0.25, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.59, 0.6, 0.75, 0.8]:
                latent_interp = alpha * encoded_latents[latent_idx_1][0][None, :] + (1 - alpha) * encoded_latents[latent_idx_2][0][None, :]
                decoded_logits = self.train.model.decode_latent(latent_interp)
                ids = torch.argmax(decoded_logits, dim=-1)
                recon_text = self.train.tokenizer.batch_decode(ids, skip_special_tokens=True)
                print(f"==> Interp w/ alpha: {alpha}:")
                print(recon_text)
                print()
                print()

            decoded_logits = self.train.model.decode_latent(encoded_latents[latent_idx_1][0][None, :])
            decoded_logits_2 = self.train.model.decode_latent(encoded_latents[latent_idx_2][0][None, :])

            ids = torch.argmax(decoded_logits, dim=-1)
            ids_2 = torch.argmax(decoded_logits_2, dim=-1)

            recon_text_1 = self.train.tokenizer.batch_decode(ids, skip_special_tokens=True)
            recon_text_2 = self.train.tokenizer.batch_decode(ids_2, skip_special_tokens=True)


            original_text = self.train.tokenizer.batch_decode(original_ids[latent_idx_1][0][None, :], skip_special_tokens=True)
            original_text_2 = self.train.tokenizer.batch_decode(original_ids[latent_idx_2][0][None, :], skip_special_tokens=True)


            print(f"==> RECON TEXT 1:\n", recon_text_1)
            print()
            print()
            print(f"==> RECON TEXT 2:\n", recon_text_2)
            print()
            print()
            print(f"==> ORIGINAL TEXT:\n", original_text)
            print()
            print()

@hydra.main(version_base="1.3", config_path="conf", config_name="train_lvae_llnl")
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().run.dir
    print(f"==> Output Dir: {output_dir}")
    trainer = VAETest(cfg)
    trainer.test_interp() 

if __name__ == "__main__":
    main() 
