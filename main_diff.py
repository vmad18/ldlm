

from diffusion.train_cfm import Trainer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="roc")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--train_bs", type=int, default=196) # 196
    parser.add_argument("--eval_bs", type=int, default=64)
    parser.add_argument("--save_and_sample_every", type=int, default=50)
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

    parser.add_argument("--train_num_steps", type=int, default=50000)
    parser.add_argument("--lr_schedule", type=str, default="linear")
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_test", type=bool, default=False)
    parser.add_argument("--num_samples", type=int, default=32)
    # parser.add_argument("--optimizer", type=str, default="adamw")
    # parser.add_argument("--adam_beta1", type=float, default=0.9)
    # parser.add_argument("--adam_beta2", type=float, default=0.999)
    # parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
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

