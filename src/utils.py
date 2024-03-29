import torch

def Patching_CLS(batch_images, patch_size, device): 
    
    batch_size = batch_images.shape[0]
    channels_num  = batch_images.shape[1]

    patch_size = patch_size
    num_patches = (batch_images.shape[2] // patch_size) * (batch_images.shape[3] // patch_size)
    patches = batch_images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, channels_num, num_patches, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)

    # # Create a CLS token (learnable parameter)
    # cls_token = torch.ones(1, 1, channels_num, patch_size, patch_size).to(device)

    # # Expand the CLS token to match the batch size and number of patches
    # cls_tokens = cls_token.expand(batch_size, 1, -1, -1, -1)

    # # Concatenate the CLS token to the patches
    # patches_with_cls = torch.cat((cls_tokens, patches), dim=1)

    # Example tensor size: [1, 1+16, 3, 8, 8]. [batch_size, num_patches, channels, patch_size, patch_size]
    return patches

if __name__ == '__main__':
    # Example usage:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_images = torch.randn(64, 3, 32, 32).to(device)  # Example image tensor
    patch_size = 8
    patches = Patching_CLS(batch_images, patch_size, device=device)
    print(patches.shape)  # Output tensor size: [64, 17, 3, 8, 8]
    print(patches[0,0,:,:,:])


