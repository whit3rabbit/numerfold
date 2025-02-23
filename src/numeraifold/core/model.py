import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NumerAIFold(nn.Module):
    """
    AlphaFold-inspired model for Numerai predictions.
    
    This model processes input features by first embedding them, adding a learnable positional
    encoding, and then passing the resulting representations through multiple transformer blocks.
    The final output is obtained by averaging the transformed embeddings and applying a linear
    projection followed by a sigmoid activation.
    
    Args:
        num_features (int): Number of input features.
        num_layers (int, optional): Number of transformer blocks. Default is 6.
        embed_dim (int, optional): Dimensionality of feature embeddings. Default is 256.
        num_heads (int, optional): Number of attention heads in the transformer blocks. Default is 8.
        ff_dim (int, optional): Dimensionality of the feed-forward network inside transformer blocks. Default is 1024.
        dropout (float, optional): Dropout rate applied across layers. Default is 0.1.
    """
    def __init__(self, num_features, num_layers=6, embed_dim=256, num_heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Embedding layer: projects each scalar feature into a higher dimensional space.
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),  # Transform each individual feature value.
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Positional encoding: learnable parameters added to the embedded features to encode sequence order.
        max_seq_len = max(num_features * 2, 1000)  # Allows for sequences longer than the number of features.
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        # Transformer blocks to capture complex feature interactions.
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output layers: normalization and projection to produce a single prediction.
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        Forward pass of the NumerAIFold model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, 1].
        
        Returns:
            tuple: 
                - torch.Tensor: Sigmoid activated predictions of shape [batch_size].
                - list: A list containing attention weights from each transformer block.
        """
        # x shape: [batch_size, seq_length, 1]
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Embed features: transform each scalar input into a vector of dimension embed_dim.
        x = self.feature_embedding(x)  # Resulting shape: [batch_size, seq_len, embed_dim]

        # Add positional encoding to preserve the order of the features.
        x = x + self.pos_encoding[:, :seq_len, :]

        # List to store attention weights from each transformer block.
        attentions = []

        # Pass through each transformer block.
        for transformer in self.transformer_blocks:
            x, attn = transformer(x)
            attentions.append(attn)

        # Normalize and average the output embeddings, then project to a single output.
        x = self.output_norm(x)
        x = x.mean(dim=1)  # Average across the sequence length.
        x = self.output_proj(x)

        # Apply sigmoid activation and squeeze the last dimension.
        return torch.sigmoid(x.squeeze(-1)), attentions


class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of a multi-head self-attention sublayer and a feed-forward network.
    
    Includes layer normalization, dropout, and residual connections for improved training stability.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feedforward = FeedForward(embed_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim].
            mask (torch.Tensor, optional): Attention mask. Default is None.
        
        Returns:
            tuple:
                - torch.Tensor: Output tensor after processing through the transformer block.
                - torch.Tensor: Attention weights from the self-attention layer.
        """
        # Apply layer normalization before self-attention.
        residual = x
        x = self.norm1(x)
        attn_output, attn_weights = self.attention(x, mask)
        # Add residual connection with dropout.
        x = residual + self.dropout(attn_output)

        # Pass through the feed-forward network.
        x = self.feedforward(x)

        return x, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    This layer allows the model to jointly attend to information from different representation subspaces.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Linear layers to compute queries, keys, and values.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # Output projection.
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for multi-head self-attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim].
            mask (torch.Tensor, optional): Attention mask. Default is None.
        
        Returns:
            tuple:
                - torch.Tensor: Output tensor after attention of shape [batch_size, seq_len, embed_dim].
                - torch.Tensor: Attention weights.
        """
        batch_size, seq_len, _ = x.size()

        # Project input and reshape for multi-head attention.
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate scaled dot-product attention scores.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided.
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Normalize the scores to probabilities.
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute the attention output.
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        # Final linear projection.
        output = self.out_proj(attn_output)
        return output, attn_weights


class FeedForward(nn.Module):
    """
    Feed-forward neural network with residual connection and layer normalization.
    
    Consists of two linear layers with a GELU activation and dropout in between.
    """
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass for the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim].
        
        Returns:
            torch.Tensor: Output tensor after applying the feed-forward network.
        """
        residual = x
        # First linear transformation followed by GELU activation.
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        # Second linear transformation.
        x = self.linear2(x)
        x = self.dropout(x)
        # Add residual connection and normalize.
        x = x + residual
        x = self.norm(x)
        return x


class ImprovedPairwiseAttention(nn.Module):
    """
    Improved Pairwise Attention module inspired by AlphaFold's Invariant Point Attention (IPA).
    
    This module models interactions between features by incorporating pairwise biases along with
    multi-head self-attention, allowing for enhanced modeling of feature relationships.
    
    Args:
        feature_dim (int): Dimensionality of feature representations.
        heads (int, optional): Number of attention heads. Default is 4.
        dropout (float, optional): Dropout rate. Default is 0.1.
    """
    def __init__(self, feature_dim, heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.heads = heads
        self.head_dim = feature_dim // heads
        assert self.head_dim * heads == feature_dim, "feature_dim must be divisible by heads"

        # Linear layers to compute queries, keys, and values.
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        # Learnable pairwise bias matrix to capture pair interactions.
        self.pair_bias = nn.Parameter(torch.zeros(feature_dim, feature_dim))

        # Output projection and layer normalization.
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        Forward pass for the improved pairwise attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_features, feature_dim].
        
        Returns:
            tuple:
                - torch.Tensor: Output tensor with pairwise interactions incorporated.
                - torch.Tensor: Attention weights.
        """
        batch_size, num_features = x.shape[0], x.shape[1]
        residual = x

        # Compute linear projections for queries, keys, and values.
        q = self.q_proj(x).view(batch_size, num_features, self.heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_features, self.heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_features, self.heads, self.head_dim)

        # Rearrange dimensions to prepare for attention calculation.
        q = q.permute(0, 2, 1, 3)  # Shape: [batch_size, heads, num_features, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Compute scaled dot-product attention scores.
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Expand pairwise bias to match dimensions and add to attention scores.
        pair_bias_expanded = self.pair_bias.unsqueeze(0).unsqueeze(0)
        pair_bias_expanded = pair_bias_expanded.expand(batch_size, self.heads, -1, -1)
        attn_weights = attn_weights + pair_bias_expanded[:, :, :num_features, :num_features]

        # Normalize the scores.
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute the weighted sum of values.
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, num_features, self.feature_dim)
        out = self.out_proj(out)

        # Apply residual connection and layer normalization.
        out = residual + self.dropout(out)
        out = self.norm(out)

        return out, attn_weights
