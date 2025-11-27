"""FlexAttention integration for hierarchical masking."""
from typing import List, Optional, Dict
import torch
from torch import Tensor
from torch.nn.attention import flex_attention
from torch.nn.attention.flex_attention import _mask_mod_signature


class HierarchicalPositions:
    """Sparse hierarchical enumeration vectors for cascading sliding windows.

    Each level has an enumeration vector [N] or [B, N]:
    - Non-zero values: sequential enumeration for that level
    - Zero values: position doesn't participate in this level

    Example (9 tokens, 4 sentences, 2 sections):
        level_enums[0] = [1,2,3,4,5,6,7,8,9]  # all tokens
        level_enums[1] = [1,0,0,2,0,3,0,0,4]  # sentence boundaries
        level_enums[2] = [1,0,0,0,0,2,0,0,0]  # section boundaries
    """

    def __init__(self, level_enums: List[Tensor]):
        """Initialize hierarchical positions.

        Args:
            level_enums: List of tensors, each of shape [N] or [B, N] where:
                - level_enums[0] contains enumeration for all tokens (level 0)
                - level_enums[1] contains enumeration for sentence-level tokens (level 1)
                - level_enums[2] contains enumeration for section-level tokens (level 2)
                - Non-zero values indicate participation in that hierarchy level
        """
        self.level_enums = level_enums
        self.num_levels = len(level_enums)
        if self.num_levels == 0:
            raise ValueError("level_enums must contain at least one level")

        # Validate all tensors have same shape (except batch dimension)
        first_enum = level_enums[0]
        first_dim = first_enum.dim()
        for i, enum in enumerate(level_enums):
            if enum.dim() != first_dim:
                raise ValueError(
                    f"All level_enums must have same number of dimensions. "
                    f"Level 0 has {first_dim} dims, level {i} has {enum.dim()} dims"
                )
            if first_dim == 1:
                if enum.shape[0] != first_enum.shape[0]:
                    raise ValueError(
                        f"All level_enums must have same sequence length. "
                        f"Level 0 has {first_enum.shape[0]}, level {i} has {enum.shape[0]}"
                    )
            elif first_dim == 2:
                if enum.shape[1] != first_enum.shape[1]:
                    raise ValueError(
                        f"All level_enums must have same sequence length. "
                        f"Level 0 has {first_enum.shape[1]}, level {i} has {enum.shape[1]}"
                    )


def generate_hierarchical_sliding_window_mask_mod(
    hierarchical_positions: HierarchicalPositions,
    active_levels: List[int],
    level_window_sizes: List[int],
) -> _mask_mod_signature:
    """Generates a hierarchical sliding window attention mask mod.

    Args:
        hierarchical_positions: HierarchicalPositions object containing enumeration vectors
        active_levels: List of hierarchy levels to use (e.g., [0] or [0,1] or [0,1,2])
        level_window_sizes: List of window sizes for each level (e.g., [256, 32, 8])
                           Must have same length as active_levels

    Returns:
        A mask_mod function with signature (b, h, q_idx, kv_idx) -> bool tensor

    Note:
        For each active level, tokens can attend to other tokens in the same level
        if their enumeration values are within that level's window_size. Masks from different
        levels are combined with OR logic (union).
    """
    if not active_levels:
        raise ValueError("active_levels must contain at least one level")
    if any(level < 0 or level >= hierarchical_positions.num_levels for level in active_levels):
        raise ValueError(
            f"active_levels must be in range [0, {hierarchical_positions.num_levels - 1}]"
        )
    if len(level_window_sizes) != len(active_levels):
        raise ValueError(
            f"level_window_sizes must have same length as active_levels. "
            f"Got {len(level_window_sizes)} window sizes for {len(active_levels)} levels"
        )

    # Extract enum tensors and window sizes for active levels
    active_enums = [hierarchical_positions.level_enums[level] for level in active_levels]
    active_window_sizes = level_window_sizes

    # Determine if we're dealing with batched or unbatched tensors
    is_batched = active_enums[0].dim() == 2

    def hierarchical_mask_mod(b, h, q_idx, kv_idx):
        """Mask mod function that implements hierarchical sliding window attention."""
        # Collect masks for each level
        level_masks = []

        for level_idx, (level_enum, window_size) in enumerate(zip(active_enums, active_window_sizes)):
            # Get enum values for query and key positions
            # Follow the same pattern as document_mask: index directly and let PyTorch broadcast
            if is_batched:
                # Handle batched case: level_enum is [B, N]
                # b is [B], q_idx/kv_idx are broadcasted by flex_attention
                q_enum = level_enum[b, q_idx]
                kv_enum = level_enum[b, kv_idx]
            else:
                # Handle unbatched case: level_enum is [N]
                q_enum = level_enum[q_idx]
                kv_enum = level_enum[kv_idx]

            # Check if both positions participate in this level (non-zero enum)
            # PyTorch will broadcast these automatically when we do operations
            q_participates = q_enum > 0
            kv_participates = kv_enum > 0

            # For level 0, use position-based windowing for consistency with simple sliding window
            # For higher levels, use enum-based windowing
            if level_idx == 0 and active_levels[0] == 0:
                # Level 0: use position-based windowing (enum values are 1-indexed positions)
                # Convert enum to position: pos = enum - 1
                q_pos = q_enum - 1
                kv_pos = kv_enum - 1
                pos_diff = torch.abs(q_pos - kv_pos)
                in_window = pos_diff < window_size
            else:
                # Higher levels: use enum-based windowing
                enum_diff = torch.abs(q_enum - kv_enum)
                in_window = enum_diff <= window_size

            # Level mask: both participate AND within window
            # PyTorch will broadcast q_participates and kv_participates automatically
            level_mask = q_participates & kv_participates & in_window
            level_masks.append(level_mask)

        # Combine all level masks with OR logic
        if len(level_masks) == 1:
            combined_mask = level_masks[0]
        else:
            # Use bitwise OR to combine masks
            combined_mask = level_masks[0]
            for mask in level_masks[1:]:
                combined_mask = combined_mask | mask

        return combined_mask

    # Set function name for identification
    levels_str = "_".join(map(str, active_levels))
    window_sizes_str = "_".join(map(str, active_window_sizes))
    hierarchical_mask_mod.__name__ = f"hierarchical_sw_levels_{levels_str}_ws_{window_sizes_str}"
    return hierarchical_mask_mod


def sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    hierarchical_positions: Optional[HierarchicalPositions] = None,
    active_levels: Optional[List[int]] = None,
    level_window_sizes: Optional[List[int]] = None,
    window_size: Optional[int] = None,
) -> torch.Tensor:
    """Wrapper around flex_attention with hierarchical masking.

    Args:
        query: (B, H, L, D) query tensor
        key: (B, H, L, D) key tensor
        value: (B, H, L, D) value tensor
        hierarchical_positions: Optional HierarchicalPositions object
        active_levels: Optional list of hierarchy levels to use
        level_window_sizes: Optional list of window sizes for each level
        window_size: Optional single window size for simple sliding window (if hierarchical_positions not provided)

    Returns:
        output: (B, H, L, D) attention output
    """
    if hierarchical_positions is not None:
        if active_levels is None or level_window_sizes is None:
            raise ValueError(
                "If hierarchical_positions is provided, active_levels and level_window_sizes must also be provided"
            )
        mask_mod = generate_hierarchical_sliding_window_mask_mod(
            hierarchical_positions, active_levels, level_window_sizes
        )
    elif window_size is not None:
        # Standard sliding window without hierarchy
        def mask_mod(b: int, h: int, q_idx: int, k_idx: int) -> bool:
            return abs(q_idx - k_idx) < window_size
    else:
        raise ValueError(
            "Either hierarchical_positions or window_size must be provided"
        )

    return flex_attention(query, key, value, mask_mod=mask_mod)


def _extract_hierarchy_window_sizes(hierarchy_config: List[Dict]) -> List[int]:
    """Extract window sizes for each hierarchy level from config.
    
    Args:
        hierarchy_config: List of dicts from config
    
    Returns:
        List of window sizes, one per hierarchy level
    """
    hierarchy_window_sizes = []
    for level_config in hierarchy_config:
        if "level_window_sizes" in level_config:
            raise NotImplementedError(
                "level_window_sizes in config not yet supported. "
                "Use window_size per hierarchy level instead."
            )
        elif "window_size" in level_config:
            hierarchy_window_sizes.append(level_config["window_size"])
        else:
            raise ValueError(
                f"window_size must be provided for hierarchy level {level_config.get('name', 'unknown')}"
            )
    return hierarchy_window_sizes


def get_hierarchy_level_attention_config(
    hierarchy_level_idx: int,
    hierarchy_config: List[Dict],
) -> tuple[List[int], List[int]]:
    """Get active levels and window sizes for a specific hierarchy level.
    
    Args:
        hierarchy_level_idx: Hierarchy level index (0-indexed, e.g., 0=raw, 1=sentence, 2=section)
        hierarchy_config: List of dicts from config
    
    Returns:
        Tuple of (active_levels, level_window_sizes) for this hierarchy level
    """
    hierarchy_window_sizes = _extract_hierarchy_window_sizes(hierarchy_config)
    
    # Active levels are [0, 1, ..., hierarchy_level_idx]
    active_levels = list(range(hierarchy_level_idx + 1))
    
    # Build level_window_sizes: each level uses its own window_size from config
    # Level 0 always uses hierarchy_window_sizes[0] (raw level)
    # Level 1 uses hierarchy_window_sizes[1] (sentence level)
    # Level 2 uses hierarchy_window_sizes[2] (section level)
    # etc.
    level_window_sizes = []
    for level_idx in active_levels:
        if level_idx < len(hierarchy_window_sizes):
            level_window_sizes.append(hierarchy_window_sizes[level_idx])
        else:
            raise ValueError(
                f"Hierarchy level {level_idx} window_size not found in config. "
                f"Only {len(hierarchy_window_sizes)} hierarchy levels defined."
            )
    
    return active_levels, level_window_sizes


def get_layer_attention_config_from_hierarchy(
    layer_idx: int,
    hierarchy_config: List[Dict],
) -> tuple[List[int], List[int]]:
    """Get active levels and window sizes for a specific layer based on hierarchy_config.
    
    This is a convenience function that extracts the attention configuration for a given
    layer index based on the hierarchy_config structure.
    
    Args:
        layer_idx: Layer index (0-indexed)
        hierarchy_config: List of dicts from config, each with:
            - name: str
            - num_layers: int
            - window_size: int (or level_window_sizes: List[int])
            - router_sparsity_target: float (optional)
    
    Returns:
        Tuple of (active_levels, level_window_sizes) for this layer
    
    Example:
        hierarchy_config = [
            {"name": "raw", "num_layers": 3, "window_size": 256},
            {"name": "sentence", "num_layers": 3, "window_size": 64},
            {"name": "section", "num_layers": 6, "window_size": 16},
        ]
        # Layer 0-2: active_levels=[0], level_window_sizes=[256]
        # Layer 3-5: active_levels=[0,1], level_window_sizes=[256, 64]
        # Layer 6-11: active_levels=[0,1,2], level_window_sizes=[256, 64, 16]
        # Note: Level 0 (tokens) always uses raw level's window_size (256)
        #       Level 1 (sentences) uses sentence level's window_size (64)
        #       Level 2 (sections) uses section level's window_size (16)
    """
    # Find which hierarchy level this layer belongs to
    current_layer = 0
    for i, level_config in enumerate(hierarchy_config):
        num_layers = level_config["num_layers"]
        if current_layer <= layer_idx < current_layer + num_layers:
            # This layer belongs to hierarchy level i
            return get_hierarchy_level_attention_config(i, hierarchy_config)
        current_layer += num_layers
    
    raise ValueError(f"layer_idx {layer_idx} is out of range for hierarchy_config")
