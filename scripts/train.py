"""Main training script for HSWT."""
import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
import torch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    # Set seed
    if "seed" in cfg:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    
    # Initialize model
    model = hydra.utils.instantiate(cfg.model)
    
    # Initialize data
    data = hydra.utils.instantiate(cfg.data)
    
    # Initialize trainer
    trainer_cfg = {
        "max_epochs": cfg.train.max_epochs,
        "accelerator": "auto",
        "devices": "auto",
    }
    trainer = L.Trainer(**trainer_cfg)
    
    # Train
    trainer.fit(model, data)


if __name__ == "__main__":
    main()

