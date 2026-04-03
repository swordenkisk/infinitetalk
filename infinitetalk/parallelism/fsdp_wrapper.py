"""FSDP Wrapper for model sharding"""
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from typing import Optional

class FSDPVideoModel:
    """
    FSDP wrapper for video generation models
    """
    
    @staticmethod
    def wrap_model(model, device_id: int, 
                   mixed_precision: str = "bf16",
                   auto_wrap_policy = None):
        """
        Wrap model with FSDP for distributed training
        
        Args:
            model: The model to wrap
            device_id: GPU device ID
            mixed_precision: "fp16", "bf16", or "fp32"
            auto_wrap_policy: Policy for auto-wrapping layers
        """
        from torch.distributed.fsdp import MixedPrecision
        
        mp_policy = {
            "fp16": MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float32
            ),
            "bf16": MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.float32
            ),
            "fp32": None
        }.get(mixed_precision)
        
        wrapped = FSDP(
            model,
            device_id=device_id,
            mixed_precision=mp_policy,
            auto_wrap_policy=auto_wrap_policy
        )
        
        return wrapped
