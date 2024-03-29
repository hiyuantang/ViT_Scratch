import torch
import torch.nn as nn
from Attention import *
from LinearHead import *
from PatchEncoder import *
from PositionalEncoding import *

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attention = Attention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear = LinearHead(dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)
    
    def forward(self, x):
        x = self.attention(x)
        x = self.linear(x)
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, patch_size, num_patches, num_blocks, embed_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.num_blocks = num_blocks
        self.max_seq_len = num_patches # Not count for CLS

        self.patch_encoder = PatchEncoder(patch_size=patch_size, embed_dim=embed_dim)
        self.PositionalEncoding = PositionalEncoding(embed_dim=embed_dim, max_seq_len=self.max_seq_len)

        # Create a list of AttentionBlock instances
        self.att_blocks = nn.ModuleList([
            AttentionBlock(embed_dim=embed_dim, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # Patch embedding and positional encoding
        x = self.patch_encoder(x)
        x = self.PositionalEncoding(x)

        # Pass the input through the attention blocks
        for att_block in self.att_blocks:
            x = att_block(x)
        return x

class RegressionViT(nn.Module):
    def __init__(self, patch_size, num_patches, num_blocks, embed_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.VisionTransformer = VisionTransformer(patch_size, num_patches, num_blocks, embed_dim, hidden_dim, num_heads, dropout)
        self.L1 = nn.Linear(embed_dim, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.L2 = nn.Linear(128, 64) 
        self.dropout2 = nn.Dropout(dropout)
        self.L3 = nn.Linear(64, 1) 
        self.activation = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.VisionTransformer(x)

        # Global average pooling
        x = x.transpose(1, 2)  # Swap dimensions to (batch_size, embed_dim, num_patches)
        x = self.global_avg_pool(x)  # Pooling along the num_patches dimension
        x = x.squeeze(-1)  # Remove the dummy dimension after pooling

        out = self.L1(x) 
        out = self.dropout1(out)
        out = self.activation(out)

        out = self.L2(out)
        out = self.dropout2(out)
        out = self.activation(out)

        out = self.L3(out)
        return out
    
class ClassificationViT(nn.Module):
    def __init__(self, num_class, patch_size, num_patches, num_blocks, embed_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.num_class = num_class
        self.VisionTransformer = VisionTransformer(patch_size, num_patches, num_blocks, embed_dim, hidden_dim, num_heads, dropout)
        self.L1 = nn.Linear(embed_dim, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.L2 = nn.Linear(128, 64) 
        self.dropout2 = nn.Dropout(dropout)
        self.L3 = nn.Linear(64, num_class) 
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.VisionTransformer(x)
        
        # Global average pooling
        x = x.transpose(1, 2)  # Swap dimensions to (batch_size, embed_dim, num_patches)
        x = self.global_avg_pool(x)  # Pooling along the num_patches dimension
        x = x.squeeze(-1)  # Remove the dummy dimension after pooling

        out = self.L1(x) # Only deal with CLS
        out = self.dropout1(out)
        out = self.activation(out)

        out = self.L2(out)
        out = self.dropout2(out)
        out = self.activation(out)

        out = self.L3(out)
        out = self.softmax(out)
        return out

if __name__ == '__main__':
    batch_image_patches = torch.randn(32, 17, 3, 8, 8)
    scaler_out = RegressionViT(patch_size=8, num_patches=16, num_blocks=2, embed_dim=128, 
                               hidden_dim=128, num_heads=8, dropout=0.)(batch_image_patches)
    
    print(scaler_out.size())

