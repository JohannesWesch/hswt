"""Visualization utilities for hierarchical attention masks."""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from pathlib import Path
from hswt.attention import (
    HierarchicalPositions, 
    generate_hierarchical_sliding_window_mask_mod,
    get_layer_attention_config_from_hierarchy,
)


def materialize_mask(mask_mod, seq_len: int, device: str = "cpu") -> np.ndarray:
    """Materialize a mask_mod function into a dense boolean matrix.
    
    Args:
        mask_mod: Mask modification function with signature (b, h, q_idx, kv_idx) -> bool tensor
        seq_len: Sequence length
        device: Device to use for computation
    
    Returns:
        Dense boolean mask of shape (seq_len, seq_len)
    """
    b = torch.tensor(0, device=device)
    h = torch.tensor(0, device=device)
    
    q_idx = torch.arange(seq_len, device=device)
    kv_idx = torch.arange(seq_len, device=device)
    
    Q, K = torch.meshgrid(q_idx, kv_idx, indexing="ij")
    dense_mask = mask_mod(b, h, Q, K)  # (seq_len, seq_len) bool
    
    # Convert to numpy for plotting
    if isinstance(dense_mask, torch.Tensor):
        dense_mask = dense_mask.cpu().numpy()
    
    return dense_mask.astype(float)


def visualize_hierarchical_attention(
    hierarchical_positions: HierarchicalPositions,
    active_levels: List[int],
    level_window_sizes: List[int],
    device: str = "cpu",
    name: Optional[str] = None,
    seq_len: Optional[int] = None,
    batch_size: int = 1,
    num_heads: int = 1,
    head_dim: int = 8,
    output_dir: Optional[str] = None,
    dpi: int = 300,
    figsize: tuple = (12, 12),
    zoom_region: Optional[tuple] = None,
):
    """Visualize the attention scores of hierarchical sliding window mask mod.

    Args:
        hierarchical_positions: HierarchicalPositions object containing enumeration vectors
        active_levels: List of hierarchy levels to use
        level_window_sizes: List of window sizes for each level
        device: Device to use for computation. Defaults to "cpu".
        name: Optional name for the visualization
        seq_len: Optional sequence length (if not provided, inferred from hierarchical_positions)
        batch_size: Batch size for visualization
        num_heads: Number of attention heads
        head_dim: Head dimension
        output_dir: Optional directory to save visualization images. Defaults to "outputs/visualizations".
        dpi: Dots per inch for the plot. Higher values = higher resolution. Default: 300.
        figsize: Figure size in inches (width, height). Default: (12, 12).
        zoom_region: Optional tuple (q_start, q_end, k_start, k_end) to create a zoomed plot.
    """
    # Infer sequence length from hierarchical positions
    if seq_len is None:
        first_enum = hierarchical_positions.level_enums[0]
        if first_enum.dim() == 1:
            seq_len = first_enum.shape[0]
        else:
            seq_len = first_enum.shape[1]

    # Generate mask
    hierarchical_mask = generate_hierarchical_sliding_window_mask_mod(
        hierarchical_positions, active_levels, level_window_sizes
    )

    # Generate name if not provided
    if name is None:
        levels_str = "_".join(map(str, active_levels))
        window_sizes_str = "_".join(map(str, level_window_sizes))
        name = f"hierarchical_sw_levels_{levels_str}_ws_{window_sizes_str}"

    # Materialize mask to dense matrix
    print(f"    Materializing mask for seq_len={seq_len}...")
    dense_mask = materialize_mask(hierarchical_mask, seq_len, device=device)

    # Set up output directory
    if output_dir is None:
        output_dir = "outputs/visualizations"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate appropriate DPI to ensure at least 1 pixel per token
    # For seq_len=8192, we want at least 8192 pixels, so with figsize=(12, 12), we need dpi >= 8192/12 ≈ 683
    min_dpi = int(np.ceil(seq_len / max(figsize)))
    actual_dpi = max(dpi, min_dpi)
    
    # Create full resolution plot
    print(f"    Creating plot (DPI: {actual_dpi}, size: {figsize})...")
    plt.figure(figsize=figsize, dpi=actual_dpi)
    plt.imshow(dense_mask, aspect="equal", interpolation="nearest", cmap="viridis")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.title(name.replace("_", " ").title())
    plt.colorbar(label="Attention Allowed")
    plt.tight_layout()
    
    output_file = output_path / f"{name}.png"
    plt.savefig(output_file, dpi=actual_dpi, bbox_inches="tight")
    plt.close()
    print(f"    Visualization saved as {output_file}")
    
    # Create zoomed plot if requested
    if zoom_region is not None:
        q_start, q_end, k_start, k_end = zoom_region
        patch = dense_mask[q_start:q_end, k_start:k_end]
        
        plt.figure(figsize=(8, 8), dpi=300)
        plt.imshow(patch, aspect="equal", interpolation="nearest", cmap="viridis")
        plt.xlabel(f"Key Tokens [{k_start}:{k_end}]")
        plt.ylabel(f"Query Tokens [{q_start}:{q_end}]")
        plt.title(f"{name.replace('_', ' ').title()} (Zoomed)")
        plt.colorbar(label="Attention Allowed")
        plt.tight_layout()
        
        zoom_file = output_path / f"{name}_zoom.png"
        plt.savefig(zoom_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Zoomed visualization saved as {zoom_file}")


def create_example_hierarchical_positions(
    seq_len: int = 200,
    hierarchy_config: Optional[List[Dict]] = None,
    boundary_positions: Optional[List[List[int]]] = None,
    device: str = "cpu",
) -> HierarchicalPositions:
    """Create example hierarchical positions for visualization.

    Args:
        seq_len: Sequence length
        hierarchy_config: Optional hierarchy config to determine number of levels.
                         If None, creates 3 levels (raw, sentence, section).
        boundary_positions: Optional list of boundary position lists.
                           Each inner list contains positions for that hierarchy level.
                           If None, creates default boundaries.
        device: Device to create tensors on

    Returns:
        HierarchicalPositions object
    """
    # Determine number of levels
    if hierarchy_config is not None:
        num_levels = len(hierarchy_config)
    else:
        num_levels = 3  # Default: raw, sentence, section
    
    # Level 0: all tokens enumerated sequentially
    level_0 = torch.arange(1, seq_len + 1, dtype=torch.int32, device=device)
    level_enums = [level_0]
    
    # Create higher levels based on boundary positions
    if boundary_positions is not None:
        if len(boundary_positions) != num_levels - 1:
            raise ValueError(
                f"boundary_positions must have {num_levels - 1} lists (one per hierarchy level above raw), "
                f"got {len(boundary_positions)}"
            )
        boundaries_list = boundary_positions
    else:
        # Create default boundaries
        boundaries_list = []
        
        # Level 1 (sentence): boundaries every 16 tokens
        sentence_spacing = 16
        sentence_boundaries = list(range(0, seq_len, sentence_spacing))
        # Ensure last position is included as a boundary if it's not already
        if sentence_boundaries[-1] != seq_len - 1:
            sentence_boundaries.append(seq_len - 1)
        boundaries_list.append(sentence_boundaries)
        
        # Level 2+ (section, etc.): boundaries every 10 sentences
        if num_levels > 2:
            for level_idx in range(2, num_levels):
                # Use sentence boundaries as base, take every 10th sentence
                section_boundaries = sentence_boundaries[::10]  # Every 10th sentence
                # Filter to ensure they're within seq_len and add last position if needed
                section_boundaries = [pos for pos in section_boundaries if pos < seq_len]
                if section_boundaries and section_boundaries[-1] != seq_len - 1:
                    section_boundaries.append(seq_len - 1)
                boundaries_list.append(section_boundaries)
    
    # Create enumeration vectors for each hierarchy level
    # Higher levels only set enum values at boundary positions (sparse)
    for level_idx, boundaries in enumerate(boundaries_list):
        level_enum = torch.zeros(seq_len, dtype=torch.int32, device=device)
        # Ensure boundaries are sorted and unique, and filter valid positions
        valid_boundaries = sorted(set(b for b in boundaries if 0 <= b < seq_len))
        for i, pos in enumerate(valid_boundaries):
            level_enum[pos] = i + 1
        level_enums.append(level_enum)
    
    return HierarchicalPositions(level_enums)


def visualize_layer_specific_hierarchical_attention(
    hierarchical_positions: HierarchicalPositions,
    layer_idx: int,
    hierarchy_config: List[Dict],
    device: str = "cpu",
    output_dir: Optional[str] = None,
):
    """Visualize hierarchical attention for a specific layer based on hierarchy_config.

    Args:
        hierarchical_positions: HierarchicalPositions object
        layer_idx: Layer index (0-indexed)
        hierarchy_config: List of dicts from config, each with:
            - name: str
            - num_layers: int
            - window_size: int (or level_window_sizes: List[int])
            - router_sparsity_target: float (optional)
        device: Device to use for computation
        output_dir: Optional directory to save visualization images. Defaults to "outputs/visualizations".
    """
    active_levels, level_window_sizes = get_layer_attention_config_from_hierarchy(
        layer_idx, hierarchy_config
    )

    visualize_hierarchical_attention(
        hierarchical_positions,
        active_levels,
        level_window_sizes,
        device=device,
        name=f"hierarchical_sliding_window_layer_{layer_idx}",
        output_dir=output_dir,
    )

