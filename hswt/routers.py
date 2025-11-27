"""Router modules for boundary detection."""
import torch
import torch.nn as nn


class Router(nn.Module):
    """Binary router for boundary detection."""
    
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Hidden dimension of input features
        """
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, hidden_dim) input features
        
        Returns:
            logits: (B, L, 2) logits for binary classification
        """
        return self.linear(x)
    
    @staticmethod
    def gumbel_binarize(
        logits: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> torch.Tensor:
        """
        Gumbel-Softmax trick for differentiable binary sampling.
        
        Args:
            logits: (B, L, 2) logits
            temperature: Temperature for Gumbel-Softmax
            hard: If True, use hard sampling (one-hot), else soft
        
        Returns:
            mask: (B, L) binary mask where 1 = boundary token
        """
        # Apply Gumbel-Softmax
        gumbel_softmax = torch.nn.functional.gumbel_softmax(
            logits, 
            tau=temperature, 
            hard=hard,
            dim=-1
        )
        # Extract boundary class (class 1)
        return gumbel_softmax[..., 1]

