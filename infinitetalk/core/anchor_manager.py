"""Anchor Manager - Sparse reference frame management"""
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class AnchorFrame:
    """Represents a single anchor frame"""
    index: int
    latent: torch.Tensor
    audio_feature: torch.Tensor
    is_keyframe: bool = False

class AnchorManager:
    """
    Manages sparse anchor frames for long video generation
    """
    
    def __init__(self, anchor_interval: int = 30, max_anchors: int = 100):
        self.anchor_interval = anchor_interval
        self.max_anchors = max_anchors
        self.anchors: Dict[int, AnchorFrame] = {}
        
    def compute_anchors(self, sequence_length: int, 
                        audio_features: torch.Tensor) -> List[int]:
        """Determine anchor positions based on regular intervals and audio variance"""
        # Regular interval anchors
        regular = list(range(0, sequence_length, self.anchor_interval))
        
        # Motion-based anchors (high variance regions)
        if audio_features.dim() > 1:
            variance = torch.var(audio_features, dim=-1)
            threshold = variance.mean() + 2 * variance.std()
            motion = torch.where(variance > threshold)[0].tolist()
        else:
            motion = []
            
        # Merge and ensure last frame is anchor
        anchors = sorted(set(regular + motion))
        if not anchors or anchors[-1] != sequence_length - 1:
            anchors.append(sequence_length - 1)
            
        return anchors[:self.max_anchors]
    
    def store_anchor(self, anchor: AnchorFrame):
        """Store an anchor frame"""
        self.anchors[anchor.index] = anchor
        
    def get_nearest_anchors(self, frame_idx: int, k: int = 2) -> List[AnchorFrame]:
        """Get k nearest anchors to frame_idx"""
        sorted_anchors = sorted(self.anchors.items(), key=lambda x: abs(x[0] - frame_idx))
        return [a for _, a in sorted_anchors[:k]]
    
    def clear(self):
        """Clear all anchors"""
        self.anchors.clear()
