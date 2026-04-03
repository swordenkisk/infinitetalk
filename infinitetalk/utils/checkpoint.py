"""Checkpoint utilities"""
import torch
import os
from typing import Dict, Optional

def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    save_dir: str,
    filename: str = None,
    extra_data: Dict = None
):
    """Save training checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if extra_data:
        checkpoint.update(extra_data)
        
    path = os.path.join(save_dir, filename)
    torch.save(checkpoint, path)
    return path

def load_checkpoint(
    model,
    checkpoint_path: str,
    optimizer=None,
    strict: bool = True
):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    epoch = checkpoint.get('epoch', 0)
    return epoch, checkpoint
