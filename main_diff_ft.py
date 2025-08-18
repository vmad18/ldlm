import sys
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from diffusion.train_cfm_ft import Trainer
import torch 

@hydra.main(config_path="conf", config_name="train_cfm_llnl", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main training script for the diffusion model, configured with Hydra.
    
    To resume training from a checkpoint, use:
    uv run main_diff.py general.checkpoint_path=/path/to/checkpoint/directory
    
    The checkpoint directory should contain either model_best.pt or model.pt
    """
    # Hydra automatically creates a unique output directory for each run
    output_dir = HydraConfig.get().run.dir
    # make the run directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Hydra output directory: {output_dir}")
    # print entire config
    print(OmegaConf.to_yaml(cfg))

    # The 'eval' flag in the config determines whether to train or evaluate.
    if cfg.general.eval:
        if not cfg.eval.path:
            raise ValueError("For evaluation, `eval.path` must be set in the config.")
        
        # crude but quick testing code
        trainer = Trainer(cfg, output_dir="./results_diff_testing")
        device = trainer.accelerator.device

        val_batch = next(trainer.val_iter)
        val_batch = {k: v.to(device) for k, v in val_batch.items()}

        bsz, s = val_batch['input_ids'].shape

        cond, trgt = val_batch['input_ids'].chunk(2, dim=-1)
        cond_mask, trgt_mask = val_batch.get('attention_mask').chunk(2, dim=-1)

        latent_cond = trainer.ae.get_latents(input_ids=cond, attn_mask=cond_mask)
        latent_trgt = trainer.ae.get_latents(input_ids=trgt, attn_mask=trgt_mask)

        with torch.no_grad():
            output_ids_list = trainer.ae.decode_latent(latent_cond)
            output_ids = torch.argmax(output_ids_list, dim=-1)

        decoded_batch = trainer.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for db in decoded_batch[:5]:
            print(">>> CONDITIONAL LATENT DECODE:", db)
            print()

        with torch.no_grad():
            output_ids_list = trainer.ae.decode_latent(latent_trgt)
            output_ids = torch.argmax(output_ids_list, dim=-1)
        
        decoded_batch = trainer.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for db in decoded_batch[:5]:
            print(">>> TARGET LATENT DECODE:", db)
            print()

        trainer.eval(latent_cond, verbose=True)

        # eval_model = Trainer.from_pretrained_for_generation(cfg.general.checkpoint_path, cfg.training.mixed_precision)
        # eval_model.eval(verbose = True)

        # Note: The from_pretrained_for_generation method is not yet converted to Hydra.
        # This path is for future implementation.
        # raise NotImplementedError("Evaluation from a checkpoint is not yet implemented with the Hydra config.")
        
    else:
        # Standard training path
        trainer = Trainer(cfg, output_dir=output_dir)
        trainer.train()

if __name__ == "__main__":
    main()

