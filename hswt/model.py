"""HSWT Lightning module."""
import pytorch_lightning as L
import torch
import torch.nn as nn
from typing import List, Optional, Dict

from hswt.attention import (
    sliding_window_attention, 
    HierarchicalPositions,
    get_hierarchy_level_attention_config,
)
from hswt.routers import Router


class RMSNorm(nn.Module):
    """RMSNorm implementation."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight


class TransformerLayer(nn.Module):
    """Single transformer layer with hierarchical attention."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window_size: int,
        active_levels: Optional[List[int]] = None,
        level_window_sizes: Optional[List[int]] = None,
        ffn_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size  # Fallback for simple sliding window
        self.active_levels = active_levels
        self.level_window_sizes = level_window_sizes
        self.head_dim = hidden_dim // num_heads
        self.ffn_dim = ffn_dim or (4 * hidden_dim)
        
        # Attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Feed-forward
        self.gate_proj = nn.Linear(hidden_dim, self.ffn_dim)
        self.up_proj = nn.Linear(hidden_dim, self.ffn_dim)
        self.down_proj = nn.Linear(self.ffn_dim, hidden_dim)
        
        # Layer norms
        # Use RMSNorm if available, otherwise use our implementation
        if hasattr(nn, 'RMSNorm'):
            self.attn_norm = nn.RMSNorm(hidden_dim)
            self.ffn_norm = nn.RMSNorm(hidden_dim)
        else:
            self.attn_norm = RMSNorm(hidden_dim)
            self.ffn_norm = RMSNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        hierarchical_positions: Optional[HierarchicalPositions] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, hidden_dim) input
            hierarchical_positions: Optional HierarchicalPositions for hierarchical attention
        
        Returns:
            output: (B, L, hidden_dim)
        """
        # Self-attention
        residual = x
        x = self.attn_norm(x)
        
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        if hierarchical_positions is not None and self.active_levels is not None:
            attn_out = sliding_window_attention(
                q, k, v,
                hierarchical_positions=hierarchical_positions,
                active_levels=self.active_levels,
                level_window_sizes=self.level_window_sizes,
            )
        else:
            # Fallback to simple sliding window
            attn_out = sliding_window_attention(
                q, k, v,
                window_size=self.window_size,
            )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.o_proj(attn_out)
        x = residual + attn_out
        
        # Feed-forward
        residual = x
        x = self.ffn_norm(x)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        ffn_out = self.down_proj(torch.nn.functional.silu(gate) * up)
        x = residual + ffn_out
        
        return x


class HSWT(L.LightningModule):
    """HSWT model with variable hierarchy levels."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_heads: int,
        hierarchy_config: List[Dict],
        router_sparsity_target: float = 0.05,
        router_sparsity_lambda: float = 0.1,
        learning_rate: float = 1e-4,
    ):
        """
        HSWT model with variable hierarchy levels.
        
        Args:
            vocab_size: Vocabulary size
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            hierarchy_config: List of dicts, each with:
                - name: str (e.g., "raw", "sentence", "section")
                - num_layers: int
                - window_size: int
                - router_sparsity_target: float (optional, per-level)
            router_sparsity_target: Default sparsity target for routers
            router_sparsity_lambda: Weight for sparsity loss
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.hierarchy_config = hierarchy_config
        self.router_sparsity_target = router_sparsity_target
        self.router_sparsity_lambda = router_sparsity_lambda
        self.learning_rate = learning_rate
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Build layers for each hierarchy level
        self.layers = nn.ModuleList()
        self.routers = nn.ModuleList()
        
        # Track which layers belong to which hierarchy level
        self.layer_to_level = []
        self.layer_active_levels = []
        self.layer_window_sizes = []
        
        for i, level_config in enumerate(hierarchy_config):
            name = level_config["name"]
            num_layers = level_config["num_layers"]
            
            # Get active levels and window sizes for this hierarchy level
            active_levels, level_window_sizes = get_hierarchy_level_attention_config(
                i, hierarchy_config
            )
            
            # Fallback window_size for simple sliding window (used if hierarchical_positions is None)
            fallback_window_size = level_window_sizes[0] if level_window_sizes else 512
            
            # Create layers for this hierarchy level
            for _ in range(num_layers):
                layer = TransformerLayer(
                    hidden_dim, 
                    num_heads, 
                    fallback_window_size,
                    active_levels=active_levels,
                    level_window_sizes=level_window_sizes,
                )
                self.layers.append(layer)
                self.layer_to_level.append(i)
                self.layer_active_levels.append(active_levels)
                self.layer_window_sizes.append(level_window_sizes)
            
            # Create router (except for first level)
            if i > 0:
                router = Router(hidden_dim)
                self.routers.append(router)
        
        # Output head
        self.ln_f = nn.RMSNorm(hidden_dim) if hasattr(nn, 'RMSNorm') else RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie weights (optional)
        # self.lm_head.weight = self.embedding.weight
    
    def _boundary_masks_to_enums(
        self, 
        boundary_masks: List[torch.Tensor], 
        seq_len: int,
        device: torch.device
    ) -> List[torch.Tensor]:
        """Convert boundary masks to enumeration vectors.
        
        Args:
            boundary_masks: List of (B, L) binary masks
            seq_len: Sequence length
            device: Device for tensors
        
        Returns:
            List of enumeration tensors, each (B, L)
        """
        B = boundary_masks[0].shape[0] if boundary_masks else 1
        enums = []
        
        # Level 0: all tokens enumerated sequentially
        level_0 = torch.arange(1, seq_len + 1, dtype=torch.int32, device=device)
        if B > 1:
            level_0 = level_0.unsqueeze(0).expand(B, -1)
        enums.append(level_0)
        
        # Higher levels: enumerate based on boundary masks
        # Only boundary positions get non-zero enum values (sparse)
        for boundary_mask in boundary_masks:
            # boundary_mask is (B, L) with 1 at boundaries
            level_enum = torch.zeros_like(boundary_mask, dtype=torch.int32)
            for b in range(B):
                boundary_positions = torch.where(boundary_mask[b] > 0)[0]
                for idx, pos in enumerate(boundary_positions):
                    level_enum[b, pos] = idx + 1
            enums.append(level_enum)
        
        return enums
    
    def _router_logits_to_enums(
        self,
        router_logits: List[torch.Tensor],
        seq_len: int,
    ) -> List[torch.Tensor]:
        """Convert router logits to enumeration vectors.
        
        Args:
            router_logits: List of (B, L, 2) router logits
            seq_len: Sequence length
        
        Returns:
            List of enumeration tensors, each (B, L)
        """
        B = router_logits[0].shape[0] if router_logits else 1
        device = router_logits[0].device
        enums = []
        
        # Level 0: all tokens enumerated sequentially
        level_0 = torch.arange(1, seq_len + 1, dtype=torch.int32, device=device)
        if B > 1:
            level_0 = level_0.unsqueeze(0).expand(B, -1)
        enums.append(level_0)
        
        # Higher levels: enumerate based on router outputs
        for router_logit in router_logits:
            # Get boundary predictions using Gumbel-Softmax
            boundary_mask = Router.gumbel_binarize(router_logit)
            level_enum = torch.zeros_like(boundary_mask, dtype=torch.int32)
            for b in range(B):
                boundary_positions = torch.where(boundary_mask[b] > 0)[0]
                for idx, pos in enumerate(boundary_positions):
                    level_enum[b, pos] = idx + 1
            enums.append(level_enum)
        
        return enums
    
    def forward(
        self,
        input_ids: torch.Tensor,
        boundary_masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: (B, L) token IDs
            boundary_masks: Optional list of (B, L) binary masks for Phase 1
        
        Returns:
            Dictionary with:
                - logits: (B, L, vocab_size)
                - router_logits: List[(B, L, 2)] - one per router
        """
        B, L = input_ids.shape
        x = self.embedding(input_ids)
        
        router_logits = []
        
        # Initialize hierarchical positions
        if boundary_masks is not None:
            # Phase 1: use provided boundary masks
            enums = self._boundary_masks_to_enums(boundary_masks, L, x.device)
            hierarchical_positions = HierarchicalPositions(enums)
        else:
            # Phase 2: start with level 0 only, will update as routers run
            level_0 = torch.arange(1, L + 1, dtype=torch.int32, device=x.device)
            if B > 1:
                level_0 = level_0.unsqueeze(0).expand(B, -1)
            hierarchical_positions = HierarchicalPositions([level_0])
        
        # Process each hierarchy level
        for i, level_config in enumerate(self.hierarchy_config):
            num_layers = level_config["num_layers"]
            
            # Apply layers for this level
            layer_start_idx = sum(self.hierarchy_config[j]["num_layers"] for j in range(i))
            for layer_offset in range(num_layers):
                layer_idx = layer_start_idx + layer_offset
                x = self.layers[layer_idx](x, hierarchical_positions=hierarchical_positions)
            
            # Apply router (except for first level)
            if i > 0:
                router_idx = i - 1
                router_logit = self.routers[router_idx](x)
                router_logits.append(router_logit)
                
                # Update hierarchical positions with new router output
                enums = self._router_logits_to_enums(router_logits, L)
                hierarchical_positions = HierarchicalPositions(enums)
        
        # Output head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return {
            "logits": logits,
            "router_logits": router_logits,
        }
    
    def training_step(self, batch, batch_idx):
        """Compute loss with auxiliary sparsity penalty."""
        input_ids = batch["input_ids"]
        boundary_masks = batch.get("boundary_masks")
        
        # Forward pass
        outputs = self.forward(input_ids, boundary_masks=boundary_masks)
        logits = outputs["logits"]
        router_logits = outputs["router_logits"]
        
        # Language modeling loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        lm_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        # Router sparsity loss
        sparsity_loss = 0.0
        for i, router_logit in enumerate(router_logits):
            # Get sparsity target for this level
            level_config = self.hierarchy_config[i + 1]
            sparsity_target = level_config.get(
                "router_sparsity_target",
                self.router_sparsity_target
            )
            
            # Compute boundary probability
            boundary_probs = torch.softmax(router_logit, dim=-1)[..., 1]
            actual_sparsity = boundary_probs.mean()
            
            # Sparsity penalty
            sparsity_loss += (actual_sparsity - sparsity_target) ** 2
        
        total_loss = lm_loss + self.router_sparsity_lambda * sparsity_loss
        
        self.log("train/lm_loss", lm_loss, on_step=True, on_epoch=True)
        self.log("train/sparsity_loss", sparsity_loss, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)
        
        return total_loss
    
    def configure_optimizers(self):
        """Return optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )
        
        # Optional: add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 10,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

