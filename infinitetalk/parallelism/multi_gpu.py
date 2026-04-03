"""Multi-GPU Manager - Production scaling"""
import torch
import torch.distributed as dist
from typing import Optional, List
import os

class MultiGPUManager:
    """
    Manages multi-GPU setup for production inference
    """
    
    def __init__(self, backend: str = "nccl"):
        self.backend = backend
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        
    def initialize(self):
        """Initialize distributed training"""
        if 'RANK' in os.environ:
            dist.init_process_group(backend=self.backend)
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(self.local_rank)
        else:
            print("Running in single-GPU mode")
            
    def is_main_process(self) -> bool:
        """Check if current process is main"""
        return self.rank == 0
    
    def barrier(self):
        """Synchronization barrier"""
        if dist.is_initialized():
            dist.barrier()
            
    def cleanup(self):
        """Clean up distributed training"""
        if dist.is_initialized():
            dist.destroy_process_group()
            
    def get_device(self) -> torch.device:
        """Get device for current process"""
        if torch.cuda.is_available():
            return torch.device(f'cuda:{self.local_rank}')
        return torch.device('cpu')
