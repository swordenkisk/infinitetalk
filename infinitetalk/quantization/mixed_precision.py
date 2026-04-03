"""Mixed Precision Manager"""
import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Optional

class MixedPrecisionManager:
    """
    Manages mixed precision training/inference
    """
    
    def __init__(self, enabled: bool = True, dtype: torch.dtype = torch.float16):
        self.enabled = enabled
        self.dtype = dtype
        self.scaler: Optional[GradScaler] = None
        
        if enabled:
            self.scaler = GradScaler()
            
    def autocast_context(self):
        """Get autocast context manager"""
        return autocast(enabled=self.enabled, dtype=self.dtype)
        
    def scale_loss(self, loss):
        """Scale loss for backward"""
        if self.scaler:
            return self.scaler.scale(loss)
        return loss
        
    def step(self, optimizer):
        """Optimizer step with gradient scaling"""
        if self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
            
    def state_dict(self):
        """Get state dict"""
        if self.scaler:
            return self.scaler.state_dict()
        return {}
        
    def load_state_dict(self, state_dict):
        """Load state dict"""
        if self.scaler and state_dict:
            self.scaler.load_state_dict(state_dict)
