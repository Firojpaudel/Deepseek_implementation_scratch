"""
This is going to be a detailed multi-head attention implementation.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #@ To avoid potential library conflicts on my system :) not that important

import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class MultiHeadAttentionDetailed(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  #@ Here we are reducing the projection dimension to match desired output dimension
        
        self.W_query= nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        self.out_proj = nn.Linear(d_out, d_out) ##@ This is the linear layer to combine the head outputs
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            "mask", 
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        keys= self.W_key(x)   # (b, num_tokens, d_out)
        queries = self.W_query(x) # (b, num_tokens, d_out)
        values = self.W_value(x)  # (b, num_tokens, d_out)
        
        # Now we split the matrix by adding num_heads dimension
        # Also, unroll the last dimension:
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # Now we need to transpose the num_tokens and num_heads dimensions
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Now we compute the attention scores with casual masking 
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        attn_scores = queries @ keys.transpose(2,3)  # Dot product for each head
        
        # Scale by sqrt(head_dim) to prevent softmax from saturating
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        
        # Apply causal mask to prevent looking at future tokens
        # mask shape: (context_length, context_length)
        # attn_scores shape: (b, num_heads, num_tokens, num_tokens)
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens].bool(), -torch.inf
        )
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        # attn_weights: (b, num_heads, num_tokens, num_tokens)
        # values: (b, num_heads, num_tokens, head_dim)
        # context_vec: (b, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values
        
        # Reshape to concatenate heads
        # (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)
        
        # Combine heads: (b, num_tokens, num_heads, head_dim) -> (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        # Apply final linear projection
        context_vec = self.out_proj(context_vec)  # (b, num_tokens, d_out)
        
        return context_vec

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Configuration
    batch_size = 2
    context_length = 8
    d_in = 512
    d_out = 512
    num_heads = 8
    dropout = 0.1
    
    print("=" * 60)
    print("Multi-Head Attention Implementation Test")
    print("=" * 60)
    
    # Create model
    mha = MultiHeadAttentionDetailed(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=dropout,
        num_heads=num_heads,
        qkv_bias=True
    )
    
    # Create sample input (representing token embeddings)
    x = torch.randn(batch_size, context_length, d_in)
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: {sum(p.numel() for p in mha.parameters()):,}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {mha.head_dim}")
    
    # Forward pass
    mha.eval()  # Set to eval mode to disable dropout for visualization
    with torch.no_grad():
        output = mha(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")
    
    # Visualize attention weights for the first batch and first head
    with torch.no_grad():
        # Get intermediate values for visualization
        queries = mha.W_query(x)
        keys = mha.W_key(x)
        
        # Reshape for multi-head
        queries = queries.view(batch_size, context_length, num_heads, mha.head_dim)
        keys = keys.view(batch_size, context_length, num_heads, mha.head_dim)
        
        # Transpose
        queries = queries.transpose(1, 2)  # (batch, heads, tokens, head_dim)
        keys = keys.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores / (mha.head_dim ** 0.5)
        
        # Apply mask
        mask = mha.mask[:context_length, :context_length].bool()
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        
        # Get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Multi-Head Attention Visualization', fontsize=16, fontweight='bold')
    
    # Plot attention weights for first 8 heads
    for head in range(min(8, num_heads)):
        row = head // 4
        col = head % 4
        
        # Get attention weights for first batch, specific head
        weights = attn_weights[0, head].cpu().numpy()
        
        sns.heatmap(
            weights,
            ax=axes[row, col],
            cmap='Blues',
            square=True,
            cbar=True,
            xticklabels=[f'T{i}' for i in range(context_length)],
            yticklabels=[f'T{i}' for i in range(context_length)],
            annot=True,
            fmt='.2f',
            annot_kws={'size': 8}
        )
        axes[row, col].set_title(f'Head {head + 1}')
        axes[row, col].set_xlabel('Key Position')
        axes[row, col].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis
    print("\n" + "=" * 60)
    print("Attention Analysis")
    print("=" * 60)
    
    # Analyze attention patterns
    mean_attention = attn_weights.mean(dim=0).mean(dim=0).cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(mean_attention, annot=True, fmt='.3f', cmap='Blues', square=True,
                xticklabels=[f'T{i}' for i in range(context_length)],
                yticklabels=[f'T{i}' for i in range(context_length)])
    plt.title('Average Attention Weights Across All Heads')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Plot attention entropy (diversity)
    plt.subplot(1, 2, 2)
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-9), dim=-1)
    mean_entropy = entropy.mean(dim=0).mean(dim=0).cpu().numpy()
    
    plt.plot(range(context_length), mean_entropy, 'o-', linewidth=2, markersize=8)
    plt.title('Attention Entropy by Position')
    plt.xlabel('Token Position')
    plt.ylabel('Entropy (bits)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Test with different sequence lengths
    print("\n" + "=" * 60)
    print("Testing Different Sequence Lengths")
    print("=" * 60)
    
    for seq_len in [4, 6, 8]:
        test_input = torch.randn(1, seq_len, d_in)
        with torch.no_grad():
            test_output = mha(test_input)
        print(f"Sequence length {seq_len}: Input {test_input.shape} -> Output {test_output.shape}")