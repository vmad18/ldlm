import numpy as np

import torch.nn.functional as F
import torch
import os 
import json

import sys

from autoencoder.train_lvae_bb import Trainer 
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="fineweb-edu_10b")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--bb", type=str, default="t5")
    
    parser.add_argument("--train_bs", type=int, default=32) # 196
    parser.add_argument("--eval_bs", type=int, default=16)
    
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--num_latents", type=int, default=128)
    parser.add_argument("--sample_every", type=int, default=50)
    parser.add_argument("--num_samples_to_gen", type=int, default=1)
    parser.add_argument("--kld_weight", type=float, default=1e-5)

    parser.add_argument("--dim_head", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=12)

    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--grad_accumulate", type=int, default=16)
    
    # parser.add_argument("--num_encoder_latents", type=int, default=32)
    # parser.add_argument("--num_decoder_latents", type=int, default=32)
    # parser.add_argument("--dim_ae", type=int, default=256)
    # parser.add_argument("--num_layers", type=int, default=2)
    # parser.add_argument("--l2_normalize_latents", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_latent_models")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_steps", type=int, default=195)
    parser.add_argument("--lr_schedule", type=str, default="linear")
    parser.add_argument("--lr_warmup_steps", type=int, default=50)
    # parser.add_argument("--optimizer", type=str, default="adamw")
    # parser.add_argument("--adam_beta1", type=float, default=0.9)
    # parser.add_argument("--adam_beta2", type=float, default=0.999)
    # parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--eval_every", type=int, default=500)
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
    parser.add_argument("--wandb_name", type=str, default="latent_ae")
    parser.add_argument(
        "--freeze_bb",
        type=str,
        default="freeze",
        choices=["freeze", "ft",],
        help=(
            "How to fine-tune LM."
        ),
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume_training", action="store_true", default=False)
    parser.add_argument("--resume_dir", type=str, default=None)
    
    # --- Arguments for Pre-computed Latents ---
    parser.add_argument("--no_precomputed_latents", action="store_false", dest="use_precomputed_latents", default=True, help="Disable using pre-computed latents for training.")
    parser.add_argument("--precomputed_latent_path", type=str, default="precomputed_latents/bigscience__T0pp_latents", help="Path to the directory of pre-computed latent .npy files.")

    args = parser.parse_args()
    print(args)

    trainer = Trainer(args)

    trainer.train() 

if __name__ == "__main__":
    main() 

