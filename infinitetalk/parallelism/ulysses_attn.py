import torch
import torch.distributed as dist
from torch import nn
from einops import rearrange

class UlyssesAttention(nn.Module):
    """
    Sequence Parallelism for long video generation
    Distributes temporal dimension across GPUs
    """
    def __init__(self, dim, num_heads, sp_group_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sp_group_size = sp_group_size
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
            
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, reference_anchors=None):
        """
        x: [B, T_local, H, W, C] where T_local = T // world_size
        """
        B, T_local, H, W, C = x.shape
        
        # Flatten spatial for attention
        x_flat = rearrange(x, 'b t h w c -> b t (h w) c')
        
        # Compute QKV
        qkv = self.qkv(x_flat)
        q, k, v = rearrange(qkv, 'b t n (three h c) -> three b h t n c', 
                           three=3, h=self.num_heads)
        
        # All-gather reference anchors if provided
        if reference_anchors is not None:
            global_anchors = self._all_gather(reference_anchors)
            # Prepend anchors to keys and values for full attention
            k = torch.cat([global_anchors, k], dim=3)
            v = torch.cat([global_anchors, v], dim=3)
            
        # Sparse attention mask (only attend to local window + anchors)
        attn_mask = self._create_sparse_mask(T_local, global_anchors.size(3) if reference_anchors else 0)
        
        # Attention
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        
        # Reshape and project
        out = rearrange(out, 'b h t n c -> b t n (h c)')
        out = self.proj(out)
        out = rearrange(out, 'b t (h w) c -> b t h w c', h=H, w=W)
        
        return out
    
    def _all_gather(self, tensor):
        """All-gather across sequence parallel group"""
        if self.world_size == 1:
            return tensor
            
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list, dim=1)  # Concatenate along time dim
    
    def _create_sparse_mask(self, seq_len, anchor_len):
        """Create mask that blocks distant future but allows anchor access"""
        total_len = seq_len + anchor_len
        mask = torch.zeros(seq_len, total_len, dtype=torch.bool)
        
        # Each position can see: all anchors + local window
        window_size = 4
        for i in range(seq_len):
            # Access to all anchors (beginning of sequence)
            mask[i, :anchor_len] = True
            # Access to local temporal window
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[i, anchor_len + start:anchor_len + end] = True
            
        return mask
