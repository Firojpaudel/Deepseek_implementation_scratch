"""
As a series to learn deepseek implementation in detail, I'm starting by trying to understand MLA. And before that I need to undestand self attention.
"""

import torch 
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias= False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
    def forward(self, x):
       keys = self.W_key(x)
       queries = self.W_query(x)
       values = self.W_value(x)
       
       # Now we need to calculate the attention scores 
       attn_scores = queries @ keys.transpose(-2, -1)
       # Then we calculate the attention weights 
       attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
       
       # Finally we have a context vector after we have the attention weights
       context_vec = attn_weights @ values
       return context_vec
   
if __name__ == "__main__":
    # Test case for SelfAttention
    torch.manual_seed(42)  # For reproducible results
    
    # Create sample input data
    batch_size, seq_len, d_in = 2, 4, 8  # 2 samples, 4 tokens, 8 input features
    d_out = 6  # 6 output features
    
    # Random input tensor
    x = torch.randn(batch_size, seq_len, d_in)
    print(f"Input shape: {x.shape}")
    
    # Initialize self-attention module
    self_attn = SelfAttention(d_in, d_out, qkv_bias=False)
    
    # Forward pass
    output = self_attn(x)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_out})")
    
    # Verify the output has correct dimensions
    assert output.shape == (batch_size, seq_len, d_out), "Output shape mismatch!"
    print("✓ Test passed! Self-attention is working correctly.")
    
    # Display first sample's output for inspection
    print(f"\nFirst sample output:\n{output[0]}")
