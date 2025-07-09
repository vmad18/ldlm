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

from diffusion.train_cfm import Trainer

@hydra.main(config_path="conf", config_name="train_cfm_llnl", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main training script for the diffusion model, configured with Hydra.
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
        
        # Note: The from_pretrained_for_generation method is not yet converted to Hydra.
        # This path is for future implementation.
        raise NotImplementedError("Evaluation from a checkpoint is not yet implemented with the Hydra config.")
        
    else:
        # Standard training path
        trainer = Trainer(cfg, output_dir=output_dir)
        trainer.train()

if __name__ == "__main__":
    main()

