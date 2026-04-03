"""Temporal Manifold - Latent interpolation utilities"""
import torch
import torch.nn as nn
from typing import Optional

class TemporalInterpolator(nn.Module):
    """Audio-guided temporal interpolation in latent space"""
    
    def __init__(self, latent_dim: int = 4, audio_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.audio_dim = audio_dim
        
        # Audio conditioning network
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim)
        )
        
    def slerp(self, z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical linear interpolation"""
        z1_norm = z1 / (torch.norm(z1, dim=-1, keepdim=True) + 1e-8)
        z2_norm = z2 / (torch.norm(z2, dim=-1, keepdim=True) + 1e-8)
        
        omega = torch.arccos(torch.clamp(
            torch.sum(z1_norm * z2_norm, dim=-1), -1, 1
        ))
        sin_omega = torch.sin(omega)
        
        if sin_omega.abs() < 1e-6:
            return (1 - t) * z1 + t * z2
        
        return (torch.sin((1-t)*omega) / sin_omega).unsqueeze(-1) * z1 + \
               (torch.sin(t*omega) / sin_omega).unsqueeze(-1) * z2
    
    def forward(self, z_start: torch.Tensor, z_end: torch.Tensor,
                steps: int, audio_feat: Optional[torch.Tensor] = None):
        """Interpolate with optional audio guidance"""
        results = []
        for i, t in enumerate(torch.linspace(0, 1, steps)):
            z_t = self.slerp(z_start, z_end, t.item())
            
            if audio_feat is not None:
                # Add audio perturbation
                audio_cond = self.audio_proj(audio_feat)
                weight = torch.sin(t * 3.14159).item()
                z_t = z_t + weight * audio_cond * 0.05
                
            results.append(z_t)
            
        return torch.stack(results)
