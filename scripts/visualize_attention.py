"""Visualize attention patterns for HSWT model based on config."""
import sys
from pathlib import Path

# Ensure project root is in path for imports
if (project_root := Path(__file__).parent.parent) not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig
import torch
from hswt.utils.visualization import (
    create_example_hierarchical_positions,
    visualize_layer_specific_hierarchical_attention,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Visualize attention patterns for all layers."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Get hierarchy config
    hierarchy_config = cfg.model.hierarchy_config
    
    # Calculate total number of layers
    total_layers = sum(level["num_layers"] for level in hierarchy_config)
    print(f"Total layers: {total_layers}")
    print(f"Hierarchy config: {hierarchy_config}")
    
    # Create example hierarchical positions
    # Use max_length from config
    seq_len = cfg.data.max_length
    
    print(f"\nCreating hierarchical positions with seq_len={seq_len}...")
    hierarchical_positions = create_example_hierarchical_positions(
        seq_len=seq_len,
        hierarchy_config=hierarchy_config,
        device=device,
    )
    
    # Set up output directory (use Hydra output dir if available, otherwise defaults/visualizations)
    if hasattr(cfg, 'hydra') and 'runtime' in cfg.hydra and 'output_dir' in cfg.hydra.runtime:
        output_dir = Path(cfg.hydra.runtime.output_dir) / "visualizations"
    else:
        output_dir = Path("outputs/visualizations")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving visualizations to: {output_dir}")
    
    # Visualize attention for each layer
    print(f"\nVisualizing attention patterns for {total_layers} layers...")
    for layer_idx in range(total_layers):
        level_name = None
        current_layer = 0
        for level_config in hierarchy_config:
            if current_layer <= layer_idx < current_layer + level_config["num_layers"]:
                level_name = level_config["name"]
                break
            current_layer += level_config["num_layers"]
        
        print(f"  Layer {layer_idx} ({level_name})...")
        try:
            visualize_layer_specific_hierarchical_attention(
                hierarchical_positions=hierarchical_positions,
                layer_idx=layer_idx,
                hierarchy_config=hierarchy_config,
                device=device,
                output_dir=str(output_dir),
            )
        except Exception as e:
            print(f"    Error visualizing layer {layer_idx}: {e}")
    
    print(f"\nVisualization complete! Images saved to: {output_dir}")


if __name__ == "__main__":
    main()

