
import numpy as np

import torch.nn.functional as F
import torch
import os 
import json

import sys

from autoencoder.train_ae import Trainer 
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="fineweb-edu_10b")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--train_bs", type=int, default=128) # 196
    parser.add_argument("--eval_bs", type=int, default=64)

    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--latent_dim", type=int, default=1536) # 196
    parser.add_argument("--num_latents", type=int, default=32)

    parser.add_argument("--dim_head", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=2048) # 196
    parser.add_argument("--num_layers", type=int, default=8)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_latent_models")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_train_steps", type=int, default=50000)
    parser.add_argument("--lr_schedule", type=str, default="linear")
    parser.add_argument("--lr_warmup_steps", type=int, default=2000)

    parser.add_argument("--eval_every", type=int, default=1000)

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--wandb_name", type=str, default="scratch_latent_vae")
    
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume_training", action="store_true", default=False)
    parser.add_argument("--resume_dir", type=str, default=None)

    args = parser.parse_args()

    trainer = Trainer(args, 
                      dataset_name = args.dataset_name, 
                      train_bs = args.train_bs, 
                      eval_bs = args.eval_bs, 
                      init_lr = args.learning_rate, 
                      train_num_steps = args.num_train_steps, 
                      lr_schedule = args.lr_schedule, 
                      num_warmup_steps = args.lr_warmup_steps, 
                      eval_every = args.eval_every, 
                      )

    trainer.train() 

if __name__ == "__main__":
    main() 

