"""
So, today's topic is causal attention. Let's start from a input tensor of any shape.
"""

import torch
import torch.nn as nn 

"""
Will implement the causal attention 2 ways. First the naive way, then the efficient way.

In Naive way, we basically convert the attention weights to a lower traiangular matrix and then divide by the sum of each row.
This basically increases the complexity where we have to perform double normalization. One for the softmax divided by sqrt(dk) and then again for the row sum.

The other way is the efficient way where we use masking to mask out the future tokens. This is done by adding a large negative number to the attention scores of the future tokens before applying softmax.
This way we only have to perform one normalization.

Also, we implement dropout after softmax to prevent overfitting. That will be in another class. Dropout might not be necessary here for simple data but to make lazy neurons fire up and prevent overfitting in large models, dropout is essential.
"""

class CausalAttentionNaive(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
    def forward(self, x):
        print(f"\n{'='*60}")
        print(f"NAIVE CAUSAL ATTENTION - STEP BY STEP")
        print(f"{'='*60}")
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        print(f"Input tensor shape: {x.shape}")
        print(f"Keys shape: {keys.shape}")
        print(f"Queries shape: {queries.shape}")
        print(f"Values shape: {values.shape}")
        
        attn_scores = queries @ keys.transpose(-2, -1)
        print(f"\nRaw attention scores (Q @ K^T):")
        print(f"Shape: {attn_scores.shape}")
        print(attn_scores)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        print(f"\nAttention weights after softmax normalization (scaled by sqrt(d_k))")
        print(f"Each row sums to 1.0")
        print(attn_weights)
        print(f"Row sums: {attn_weights.sum(dim=1)}")
        
        # Now creating the elements above the diagonal to be zero
        context_length = attn_scores.shape[0]
        
        mask_simple = torch.tril(torch.ones(context_length, context_length))
        print(f"\nLOWER TRIANGULAR MASK (Naive Approach):")
        print(f"This mask keeps only past and current tokens (1s), zeros out future tokens (0s)")
        print(f"Lower triangular matrix (includes diagonal):")
        print(mask_simple)

        print(f"\n{'='*50} APPLYING MASK TO ATTENTION WEIGHTS {'='*50}")
        masked_simple = attn_weights * mask_simple
        print(f"Attention weights after masking (future tokens = 0):")
        print(masked_simple)
        
        print(f"\nPROBLEM: Rows don't sum to 1 anymore after masking!")
        rows_sum = masked_simple.sum(dim=1, keepdim=True)
        print(f"Current row sums: {rows_sum.squeeze()}")
        
        normalized_masked_simple = masked_simple / rows_sum
        print(f"\nRE-NORMALIZED attention weights (rows sum to 1 again):")
        print(f"This is why naive approach requires DOUBLE normalization!")
        print(normalized_masked_simple)
        print(f"Verification - Row sums: {normalized_masked_simple.sum(dim=1)}")
        
        context_vec = normalized_masked_simple @ values
        print(f"\nFinal context vectors (Attention @ Values):")
        print(f"Shape: {context_vec.shape}")
        
        return context_vec

# Efficient Causal Attention
class CausalAttentionEfficient(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
    def forward(self, x):
        print(f"\n{'='*60}")
        print(f"EFFICIENT CAUSAL ATTENTION - STEP BY STEP")
        print(f"{'='*60}")
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        print(f"Input tensor shape: {x.shape}")
        print(f"Keys shape: {keys.shape}")
        print(f"Queries shape: {queries.shape}")
        print(f"Values shape: {values.shape}")
        
        attn_scores = queries @ keys.transpose(-2, -1)
        print(f"\nRaw attention scores (Q @ K^T):")
        print(f"Shape: {attn_scores.shape}")
        print(attn_scores)
        
        # Creating upper triangular mask to mask out future tokens
        context_length = attn_scores.shape[0]
        mask_efficient = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        print(f"\nUPPER TRIANGULAR MASK (Efficient Approach):")
        print(f"This mask identifies future token positions (1s) to be masked out")
        print(f"Upper triangular matrix (excludes diagonal):")
        print(mask_efficient)
        
        # Using a large negative number to mask out the future tokens
        masked = attn_scores.masked_fill(mask_efficient.bool(), -torch.inf)
        print(f"\n{'='*50} ATTENTION SCORES AFTER MASKING {'='*50}")
        print(f"Future positions filled with -inf (will become 0 after softmax)")
        print(f"This ensures future tokens get 0 attention weight!")
        print(masked)
        
        print(f"\nAPPLYING SOFTMAX TO MASKED SCORES (with sqrt(d_k) scaling)")
        print(f"Key advantage: Only ONE normalization step needed!")
        print(f"-inf values become 0 after softmax automatically")
        attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
        print(f"\nFinal attention weights (rows sum to 1 automatically):")
        print(attn_weights)
        print(f"Verification - Row sums: {attn_weights.sum(dim=1)}")
        
        context_vec = attn_weights @ values
        print(f"\nFinal context vectors (Attention @ Values):")
        print(f"Shape: {context_vec.shape}")
        
        return context_vec

class CausalAttentionWithDropout(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False, dropout_prob=0.5):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        print(f"\n{'='*60}")
        print(f"CAUSAL ATTENTION WITH DROPOUT - STEP BY STEP")
        print(f"{'='*60}")
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        print(f"Input tensor shape: {x.shape}")
        print(f"Keys shape: {keys.shape}")
        print(f"Queries shape: {queries.shape}")
        print(f"Values shape: {values.shape}")
        
        attn_scores = queries @ keys.transpose(-2, -1)
        print(f"\nRaw attention scores (Q @ K^T):")
        print(f"Shape: {attn_scores.shape}")
        print(attn_scores)
        
        context_length = attn_scores.shape[0]
        mask_efficient = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        print(f"\nUPPER TRIANGULAR MASK (Same as efficient approach):")
        print(f"This mask identifies future token positions (1s) to be masked out")
        print(mask_efficient)
        
        masked = attn_scores.masked_fill(mask_efficient.bool(), -torch.inf)
        print(f"\n{'='*50} ATTENTION SCORES AFTER MASKING {'='*50}")
        print(f"Future positions filled with -inf:")
        print(masked)
        
        attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
        print(f"\nAttention weights BEFORE dropout:")
        print(attn_weights)
        print(f"Row sums before dropout: {attn_weights.sum(dim=1)}")
        
        # Applying dropout to attention weights
        print(f"\n{'='*50} APPLYING DROPOUT {'='*50}")
        print(f"Dropout probability: {self.dropout.p}")
        print(f"During training: randomly zeros out {self.dropout.p*100:.1f}% of attention weights")
        print(f"Remaining weights are scaled by 1/{1-self.dropout.p:.2f} to maintain expected sum")
        
        attn_weights_dropped = self.dropout(attn_weights)
        print(f"\nAttention weights AFTER dropout:")
        print(attn_weights_dropped)
        print(f"Row sums after dropout: {attn_weights_dropped.sum(dim=1)}")
        print(f"Note: Sums may differ from 1.0 due to dropout during training")
        
        context_vec = attn_weights_dropped @ values
        print(f"\nFinal context vectors (Dropped_Attention @ Values):")
        print(f"Shape: {context_vec.shape}")
        
        return context_vec
    
if __name__ == "__main__":
    # Test case for CausalAttentionNaive
    torch.manual_seed(42)  # For reproducible results
    
    # Create sample input data
    context_length, d_in = 4, 8  # 4 tokens, 8 input features
    d_out = 6  # 6 output features
    
    # Random input tensor
    x = torch.randn(context_length, d_in)
    print(f"Input shape: {x.shape}")
    
    # Initialize causal attention module (naive)
    causal_attn_naive = CausalAttentionNaive(d_in, d_out, qkv_bias=False)
    
    # Forward pass
    print("\n" + "="*20 + " Naive Causal Attention " + "="*20 + "\n")
    output_naive = causal_attn_naive(x)
    print(f"Output shape (Naive): {output_naive.shape}")
    print(f"Expected shape: ({context_length}, {d_out})")
    
    # Verify the output has correct dimensions
    assert output_naive.shape == (context_length, d_out), "Output shape mismatch!"
    print("Test passed! Naive Causal attention is working correctly.")
    
    # Display output for inspection
    print(f"\nNaive Causal Attention output:\n{output_naive}")
    
    # Test case for CausalAttentionEfficient
    # Initialize causal attention module (efficient)
    causal_attn_efficient = CausalAttentionEfficient(d_in, d_out, qkv_bias=False)
    
    # IMPORTANT: Copy weights from naive to efficient for fair comparison
    print(f"\nCopying weights from naive to efficient implementation for fair comparison...")
    causal_attn_efficient.W_query.weight.data = causal_attn_naive.W_query.weight.data.clone()
    causal_attn_efficient.W_key.weight.data = causal_attn_naive.W_key.weight.data.clone()
    causal_attn_efficient.W_value.weight.data = causal_attn_naive.W_value.weight.data.clone()
    
    # Forward pass
    print("\n" + "="*20 + " Efficient Causal Attention " + "="*20 + "\n")
    output_efficient = causal_attn_efficient(x)
    print(f"Output shape (Efficient): {output_efficient.shape}")
    print(f"Expected shape: ({context_length}, {d_out})")
    
    # Verify the output has correct dimensions
    assert output_efficient.shape == (context_length, d_out), "Output shape mismatch!"
    print("Test passed! Efficient Causal attention is working correctly.")
    
    # Display output for inspection
    print(f"\nEfficient Causal Attention output:\n{output_efficient}")
    
    # Compare outputs
    print(f"\n{'='*60}")
    print(f"COMPARING OUTPUTS")
    print(f"{'='*60}")
    print(f"Outputs are identical: {torch.allclose(output_naive, output_efficient, atol=1e-6)}")
    print(f"Maximum difference: {torch.max(torch.abs(output_naive - output_efficient)).item()}")
    
    # Test case for CausalAttentionWithDropout
    # Initialize causal attention module (with dropout)
    causal_attn_dropout = CausalAttentionWithDropout(d_in, d_out, qkv_bias=False, dropout_prob=0.3)
    
    # IMPORTANT: Copy weights from naive to dropout version for fair comparison
    print(f"\nCopying weights from naive to dropout implementation for fair comparison...")
    causal_attn_dropout.W_query.weight.data = causal_attn_naive.W_query.weight.data.clone()
    causal_attn_dropout.W_key.weight.data = causal_attn_naive.W_key.weight.data.clone()
    causal_attn_dropout.W_value.weight.data = causal_attn_naive.W_value.weight.data.clone()
    
    # Forward pass
    print("\n" + "="*20 + " Causal Attention with Dropout " + "="*20 + "\n")
    
    # Set to training mode to see dropout effect
    causal_attn_dropout.train()
    output_dropout = causal_attn_dropout(x)
    print(f"Output shape (With Dropout): {output_dropout.shape}")
    print(f"Expected shape: ({context_length}, {d_out})")
    
    # Verify the output has correct dimensions
    assert output_dropout.shape == (context_length, d_out), "Output shape mismatch!"
    print("Test passed! Causal attention with dropout is working correctly.")
    
    # Display output for inspection
    print(f"\nCausal Attention with Dropout output:\n{output_dropout}")
    
    # Test in evaluation mode (no dropout)
    print(f"\n{'='*60}")
    print(f"TESTING IN EVALUATION MODE (NO DROPOUT)")
    print(f"{'='*60}")
    causal_attn_dropout.eval()
    with torch.no_grad():
        output_dropout_eval = causal_attn_dropout(x)
    
    print(f"Outputs identical in eval mode: {torch.allclose(output_efficient, output_dropout_eval, atol=1e-6)}")
    print(f"Max difference in eval mode: {torch.max(torch.abs(output_efficient - output_dropout_eval)).item()}")