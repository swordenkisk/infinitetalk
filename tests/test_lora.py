"""Tests for LoRA functionality"""
import pytest
import torch
from infinitetalk.models.lora_manager import DualLoRAManager

class MockModel:
    """Mock model for testing"""
    def __init__(self):
        self.linear = torch.nn.Linear(10, 10)
        
def test_lora_manager_init():
    """Test LoRA manager initialization"""
    model = MockModel()
    manager = DualLoRAManager(model)
    
    assert manager.base_model == model
    assert len(manager.loaded_loras) == 0

def test_lora_loading():
    """Test LoRA loading"""
    model = MockModel()
    manager = DualLoRAManager(model)
    
    # Mock load (would need actual LoRA files for real test)
    # manager.load_lora("test", "path/to/lora", scale=0.8)
    # assert "test" in manager.loaded_loras

def test_set_active():
    """Test setting active adapters"""
    model = MockModel()
    manager = DualLoRAManager(model)
    
    # Mock setup
    manager.loaded_loras = {"lora1": "path1", "lora2": "path2"}
    manager.set_active(["lora1", "lora2"], [0.8, 0.6])
    
    assert manager.active_adapters == ["lora1", "lora2"]
    assert manager.adapter_weights["lora1"] == 0.8
    assert manager.adapter_weights["lora2"] == 0.6
