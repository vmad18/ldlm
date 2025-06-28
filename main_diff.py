import sys
import os
import argparse

# Add the project root to the Python path to allow absolute imports
# This allows us to run this script from within the ldlm directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ldlm.diffusion.train_cfm import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="fineweb-edu_10b")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--eval_bs", type=int, default=8)
    parser.add_argument("--save_and_sample_every", type=int, default=5000)
    parser.add_argument("--ema_update_every", type=int, default=10)

    # parser.add_argument("--num_encoder_latents", type=int, default=32)
    # parser.add_argument("--num_decoder_latents", type=int, default=32)
    # parser.add_argument("--dim_ae", type=int, default=256)
    # parser.add_argument("--num_layers", type=int, default=2)
    # parser.add_argument("--l2_normalize_latents", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./diff_results")
    parser.add_argument("--latent_model_path", type=str, default="./saved_latent_models/roc/2025-06-11_20-38-53/")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--model_dim", type=float, default=1024)
    parser.add_argument("--num_layers", type=float, default=10)

    # Arguments for dataset sharding and pre-computed latents
    parser.add_argument("--use_precomputed_latents", action="store_true", help="Use pre-computed latents for training.")
    parser.add_argument("--precomputed_latent_path", type=str, default=None, help="Path to the directory with pre-computed latents and metadata.")
    parser.add_argument("--shard_size", type=int, default=None, help="Use a smaller shard of the dataset for training/testing.")

    parser.add_argument("--train_num_steps", type=int, default=6250)
    parser.add_argument("--lr_schedule", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--eval_test", type=bool, default=False)
    parser.add_argument("--num_samples", type=int, default=32)
    # parser.add_argument("--optimizer", type=str, default="adamw")
    # parser.add_argument("--adam_beta1", type=float, default=0.9)
    # parser.add_argument("--adam_beta2", type=float, default=0.999)
    # parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--eval_every", type=int, default=10000)
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
    parser.add_argument("--wandb_name", type=str, default="cfm_model")
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

    args = parser.parse_args()

    trainer = Trainer(args,
                      gradient_accumulate_every=4,
                      init_lr = args.learning_rate,
                      ema_decay = 0.995,
                      adam_betas=(0.9, 0.99),
                      adam_weight_decay=0.01,)

    trainer.train()

if __name__ == "__main__":
    main()

