"""Tests for sparse frame generation"""
import pytest
import torch
from infinitetalk.core.sparse_frame_generator import SparseFrameGenerator, SparseAnchorConfig
from infinitetalk.core.anchor_manager import AnchorManager

def test_sparse_anchor_config():
    """Test anchor configuration"""
    config = SparseAnchorConfig(
        anchor_interval=30,
        identity_lock_strength=0.95,
        color_consistency_weight=0.8
    )
    assert config.anchor_interval == 30
    assert config.identity_lock_strength == 0.95

def test_anchor_manager():
    """Test anchor manager"""
    manager = AnchorManager(anchor_interval=30)
    
    # Test anchor computation
    audio = torch.randn(1, 100, 512)
    anchors = manager.compute_anchors(100, audio)
    
    assert len(anchors) > 0
    assert anchors[-1] == 99  # Last frame should be anchor
    
def test_slerp_interpolation():
    """Test spherical linear interpolation"""
    from infinitetalk.core.temporal_manifold import TemporalInterpolator
    
    interpolator = TemporalInterpolator(latent_dim=4)
    
    z1 = torch.randn(1, 4, 8, 8)
    z2 = torch.randn(1, 4, 8, 8)
    
    result = interpolator.slerp(z1, z2, 0.5)
    assert result.shape == z1.shape

def test_consistency_lock():
    """Test consistency lock"""
    from infinitetalk.core.consistency_lock import ConsistencyLock
    
    lock = ConsistencyLock(identity_weight=0.95, color_weight=0.8)
    
    latent = torch.randn(1, 4, 8, 8)
    ref_latent = torch.randn(1, 4, 8, 8)
    
    locked = lock.apply_lock(latent, ref_latent, strength=1.0)
    assert locked.shape == latent.shape
