"""Attention Processor - Custom attention mechanisms"""
import torch
import torch.nn as nn
from typing import Optional

class UlyssesAttentionProcessor(nn.Module):
    """
    Attention processor for sequence parallelism
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(hidden_size, hidden_size)
        self.to_v = nn.Linear(hidden_size, hidden_size)
        self.to_out = nn.Linear(hidden_size, hidden_size)
        
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, 
                 attention_mask=None, **kwargs):
        """Process attention with optional sequence parallelism"""
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = self.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        hidden_states = torch.matmul(attn_weights, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        hidden_states = self.to_out(hidden_states)
        
        return hidden_states
