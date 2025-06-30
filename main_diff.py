import sys
import os
import argparse
from pathlib import Path

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
    parser.add_argument("--save_and_sample_every", type=int, default=500)
    parser.add_argument("--ema_update_every", type=int, default=10)

    # parser.add_argument("--num_encoder_latents", type=int, default=32)
    # parser.add_argument("--num_decoder_latents", type=int, default=32)
    # parser.add_argument("--dim_ae", type=int, default=256)
    # parser.add_argument("--num_layers", type=int, default=2)
    # parser.add_argument("--l2_normalize_latents", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./diff_results")
    parser.add_argument("--latent_model_path", type=str, default="./saved_latent_models/fineweb-edu_10b/2025-06-28_17-24-42")
    parser.add_argument("--vae_cfg_path", type=str, default=None, help="Path to the VAE's configuration file (e.g., vae_args.json). Required for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--model_dim", type=float, default=1024)
    parser.add_argument("--num_layers", type=float, default=10)
    parser.add_argument("--model_name", type=str, default="bigscience__T0pp")

    # Arguments for dataset sharding and pre-computed latents
    parser.add_argument("--use_precomputed_latents", action="store_true", help="Use pre-computed latents for training.")
    parser.add_argument("--precomputed_latent_path", type=str, default="./precomputed_latents/bigscience__T0pp_latents", help="Path to the directory with pre-computed latents and metadata.")
    parser.add_argument("--shard_size", type=int, default=None, help="Use a smaller shard of the dataset for training/testing.")

    parser.add_argument("--train_num_steps", type=int, default=6250)
    parser.add_argument("--lr_schedule", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--eval_test", type=bool, default=False)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--grad_accumulate", type=int, default=4)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
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

    # Evaluation arguments
    parser.add_argument("--eval_from_path", type=str, default=None, help="Path to the model checkpoint (.pt file) for evaluation.")
    parser.add_argument("--num_gen_samples", type=int, default=5, help="Number of samples to generate during evaluation.")
    parser.add_argument("--gen_steps", type=int, default=100, help="Number of steps for the diffusion sampler.")
    parser.add_argument("--gen_batch_size", type=int, default=5, help="Batch size for generation.")
    parser.add_argument("--gen_max_length", type=int, default=256, help="Max sequence length for the generated text.")

    args = parser.parse_args()

    if args.eval:
        if not args.eval_from_path:
            raise ValueError("Must provide --eval_from_path for evaluation.")
        
        # Use the new classmethod for a clean, generation-focused setup
        # The path should be to the directory containing the checkpoints
        checkpoint_dir = Path(args.eval_from_path).parent if Path(args.eval_from_path).is_file() else Path(args.eval_from_path)
        trainer = Trainer.from_pretrained_for_generation(checkpoint_dir)
        
        trainer.eval(
            batch_size=args.gen_batch_size,
            gen_mult=args.gen_steps,
            max_gen_length=args.gen_max_length,
            num_samples=args.num_gen_samples,
        )
    else:
        # This path now calls the clean, training-only __init__
        trainer = Trainer(args
                      )
        trainer.train()

if __name__ == "__main__":
    main()

