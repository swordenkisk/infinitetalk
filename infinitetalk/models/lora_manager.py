"""LoRA Manager - Dual LoRA loading and management"""
import torch
from peft import PeftModel
from typing import Dict, List, Optional

class DualLoRAManager:
    """
    Manages dual LoRA system:
    - FusioniX: Temporal consistency
    - LightX2V: 4-step inference speed
    """
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.loaded_loras: Dict[str, str] = {}
        self.active_adapters: List[str] = []
        self.adapter_weights: Dict[str, float] = {}
        
    def load_lora(self, name: str, path: str, scale: float = 1.0):
        """Load a LoRA adapter"""
        if name not in self.loaded_loras:
            if hasattr(self.base_model, 'load_adapter'):
                self.base_model.load_adapter(path, adapter_name=name)
            else:
                self.base_model = PeftModel.from_pretrained(
                    self.base_model, path, adapter_name=name
                )
            self.loaded_loras[name] = path
            
        self.adapter_weights[name] = scale
        
    def set_active(self, adapters: List[str], weights: Optional[List[float]] = None):
        """Set active adapters with weights"""
        if weights is None:
            weights = [self.adapter_weights.get(a, 1.0) for a in adapters]
            
        self.active_adapters = adapters
        for adapter, weight in zip(adapters, weights):
            self.adapter_weights[adapter] = weight
            
        if hasattr(self.base_model, 'set_adapters'):
            self.base_model.set_adapters(adapters, weights)
            
    def unload_lora(self, name: str):
        """Unload a specific LoRA"""
        if name in self.loaded_loras:
            if hasattr(self.base_model, 'delete_adapter'):
                self.base_model.delete_adapter(name)
            del self.loaded_loras[name]
            if name in self.adapter_weights:
                del self.adapter_weights[name]
                
    def get_active_config(self) -> Dict:
        """Get current LoRA configuration"""
        return {
            'active_adapters': self.active_adapters,
            'weights': {a: self.adapter_weights.get(a, 1.0) for a in self.active_adapters},
            'loaded': list(self.loaded_loras.keys())
        }
