import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super().__init__()
        self.embed_dim = embed_dim

        # Create positional encodings (sine and cosine functions)
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer (non-learnable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encodings to the input embeddings
        return x + self.pe

# Example usage:
if __name__ == '__main__':
    batch_size, num_patches, embed_dim = 32, 16, 128
    max_seq_len = num_patches + 1  # Include the CLS token

    # Create patch embeddings (replace with your actual patch embeddings)
    patch_embeddings = torch.randn(batch_size, max_seq_len, embed_dim)

    # Initialize positional encoding
    positional_encoding = PositionalEncoding(embed_dim, max_seq_len)

    # Add positional encoding to patch embeddings
    patch_embeddings_with_pos = positional_encoding(patch_embeddings)

    print("Patch embeddings with positional encoding shape:", patch_embeddings_with_pos.shape)
