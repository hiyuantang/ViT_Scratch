import torch
import torch.nn as nn

class PatchEncoder(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.projection1 = nn.Linear(patch_size * patch_size * 3, embed_dim)
        self.activation = nn.ReLU()
        self.projection2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, patch_num, channels, _, _ = x.size()
        x = x.reshape(batch_size, patch_num, self.patch_size * self.patch_size * channels)
        x = self.projection1(x)
        x = self.activation(x)
        x = self.projection2(x)
        return x  # Output size: [batch_size, num_patches, embed_dim]

if __name__ == "__main__": 
    patch_encoder = PatchEncoder(patch_size=8, embed_dim=128)
    image = torch.randn(32, 17, 3, 8, 8)
    patches = patch_encoder(image)
    # Print the shape of the patches
    print(patches.shape)