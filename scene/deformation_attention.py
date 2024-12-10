import torch
import torch.nn as nn
class AttentionFeatureOut(nn.Module):
    def __init__(self, W=32, D=8, num_heads=8, dropout=0.1):  # Change W to 128
        super(AttentionFeatureOut, self).__init__()

        # Define the multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=W, num_heads=num_heads, dropout=dropout)

        # Define a linear layer to project the output of the attention to the desired feature size
        self.linear_out = nn.Linear(W, W)

        # Define normalization and dropout layers for better training stability
        self.layer_norm = nn.LayerNorm(W)
        self.dropout = nn.Dropout(dropout)

        self.D = D
        self.W = W

    def forward(self, x):
        """
        Forward pass for attention-based feature processing.
        - x: Input tensor of shape [batch_size, seq_len, W], where W is the feature size.
        """
        # Attention expects input of shape [seq_len, batch_size, feature_size], so we transpose the input
        x = x.transpose(0, 1)  # Shape: [seq_len, batch_size, W]

        # Apply multi-head attention
        attn_output, attn_output_weights = self.attention(x, x, x)

        # After attention, we add residual connection and apply layer normalization
        x = x + attn_output
        x = self.layer_norm(x)

        # Apply a linear transformation to the output
        output = self.linear_out(x)

        # Apply dropout for regularization
        output = self.dropout(output)

        # Transpose back to original shape [batch_size, seq_len, W]
        output = output.transpose(0, 1)

        return output


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()

        # Multihead attention layer
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Linear layer for output projection (can adjust to desired output dimensions)
        self.projection = nn.Linear(embed_dim, embed_dim)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for self-attention mechanism.
        - x: Input tensor of shape [batch_size, seq_len, embed_dim]
        """

        # We need to transpose to shape [seq_len, batch_size, embed_dim] for multihead attention
        x = x.transpose(0, 1)  # Shape: [seq_len, batch_size, embed_dim]

        # Apply multi-head attention
        attn_output, attn_output_weights = self.attention(x, x, x)  # Self-attention

        # Apply residual connection and layer normalization
        x = x + attn_output  # Add residual connection
        x = self.layer_norm(x)  # Apply layer normalization

        # Project the output through a linear layer
        x = self.projection(x)

        # Apply dropout
        x = self.dropout(x)

        # Transpose back to [batch_size, seq_len, embed_dim]
        x = x.transpose(0, 1)

        return x

"""import torch
import torch.nn as nn

class AttentionFeatureOut(nn.Module):
    def __init__(self, W=32, D=8, num_heads=8, dropout=0.1):
        super(AttentionFeatureOut, self).__init__()

        # Define the linear projection from W=32 to 128
        self.projection = nn.Linear(W, 128)  # W=32 to 128

        # Define the multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=num_heads, dropout=dropout)

        # Define a linear layer to project the output of the attention to the desired feature size
        self.linear_out = nn.Linear(128, W)

        # Define normalization and dropout layers
        self.layer_norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(dropout)

        self.D = D
        self.W = W

    def forward(self, x):
 
        print(f"Shape before projection: {x.shape}")  # Should be [batch_size, seq_len, 32]

        # Flatten [batch_size, seq_len, input_features] to [batch_size * seq_len, input_features]
        batch_size, seq_len, input_features = x.shape
        x = x.view(batch_size * seq_len, input_features)  # Shape: [batch_size * seq_len, 32]

        # Apply the projection (Linear layer) to increase feature size from 32 to 128
        x = self.projection(x)  # Shape: [batch_size * seq_len, 128]

        # Reshape back to [batch_size, seq_len, 128]
        x = x.view(batch_size, seq_len, -1)  # Shape: [batch_size, seq_len, 128]

        print(f"Shape after projection: {x.shape}")  # Should be [batch_size, seq_len, 128]

        # Attention expects input of shape [seq_len, batch_size, 128]
        x = x.transpose(0, 1)  # Shape: [seq_len, batch_size, 128]

        print(f"Shape before attention: {x.shape}")  # Should be [seq_len, batch_size, 128]

        # Apply multi-head attention
        attn_output, attn_output_weights = self.attention(x, x, x)

        # After attention, add residual connection and apply layer normalization
        x = x + attn_output
        x = self.layer_norm(x)

        # Apply a linear transformation to the output
        output = self.linear_out(x)

        # Apply dropout for regularization
        output = self.dropout(output)

        # Transpose back to the original shape [batch_size, seq_len, W]
        output = output.transpose(0, 1)

        return output
"""