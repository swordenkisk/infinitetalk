"""Consistency Lock - Identity and color consistency enforcement"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyLock(nn.Module):
    """
    Maintains identity and color consistency across frames
    using reference image anchoring
    """
    
    def __init__(self, identity_weight: float = 0.95, color_weight: float = 0.8):
        super().__init__()
        self.identity_weight = identity_weight
        self.color_weight = color_weight
        
    def compute_identity_loss(self, generated: torch.Tensor, 
                              reference: torch.Tensor,
                              identity_encoder) -> torch.Tensor:
        """L2 loss on identity embeddings"""
        gen_id = identity_encoder(generated)
        ref_id = identity_encoder(reference)
        return F.mse_loss(gen_id, ref_id)
    
    def compute_color_loss(self, generated: torch.Tensor,
                          reference: torch.Tensor) -> torch.Tensor:
        """Histogram matching loss for color consistency"""
        # Mean color statistics
        gen_mean = generated.mean(dim=[-2, -1])
        ref_mean = reference.mean(dim=[-2, -1])
        
        gen_std = generated.std(dim=[-2, -1])
        ref_std = reference.std(dim=[-2, -1])
        
        mean_loss = F.mse_loss(gen_mean, ref_mean)
        std_loss = F.mse_loss(gen_std, ref_std)
        
        return mean_loss + std_loss
    
    def apply_lock(self, latent: torch.Tensor, ref_latent: torch.Tensor,
                   strength: float = 1.0) -> torch.Tensor:
        """Blend latent towards reference for consistency"""
        alpha = self.identity_weight * strength
        return alpha * ref_latent + (1 - alpha) * latent
