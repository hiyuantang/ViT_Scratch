import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        att_output, _ = self.att(q, k, v)
        att_output = self.dropout(att_output)
        att_output = att_output + x  # Residual connection
        att_output = self.layer_norm(att_output)
        return att_output
    
if __name__ == '__main__':
    batch_size, num_patches, embed_dim = 32, 16, 128
    max_seq_len = num_patches + 1  # Include the CLS token

    # Create patch embeddings (replace with your actual patch embeddings)
    patch_embeddings_pos = torch.randn(batch_size, max_seq_len, embed_dim)
    att_output = Attention(embed_dim, num_heads=8, dropout=0.)(patch_embeddings_pos)
    print(att_output.size())