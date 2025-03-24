import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

import vae_SPADE as vae
from deform_images import load_mhd_image, elastic_transform_3d, save_mhd_image, save_mhd_image_without_metadata

# To ensure reproducible training/validation split
random.seed(41)

Z_DIM = 256

# Find out if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define transformations: Center crop to 256x256, then resize to 64x64
img_size = 64
transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.Resize(img_size)
])

def plot_slices(mask_tensor, synthetic_tensor, num_slices=5, alpha=0.4):
    """
    Plot `num_slices` pairs of input masks and generated synthetic images,
    including an overlay visualization.

    Parameters:
        - mask_tensor: torch.Tensor, shape (D, H, W) - The input mask
        - synthetic_tensor: torch.Tensor, shape (D, H, W) - The synthetic image
        - num_slices: int - Number of slices to visualize
        - alpha: float - Transparency level for the mask overlay (0 = fully transparent, 1 = fully opaque)
    """
    fig, axes = plt.subplots(num_slices, 3, figsize=(12, 2 * num_slices))

    depth = mask_tensor.shape[0]  # Number of slices in the 3D volume
    indices = torch.linspace(0, depth - 1, num_slices).long()  # Select evenly spaced slices

    for i, idx in enumerate(indices):
        # Extract the slices
        mask_tensor = torch.tensor(mask_tensor, dtype=torch.float32)  # Convert to PyTorch tensor if needed
        mask_slice = mask_tensor[idx].cpu()

        # Apply transformations before plotting
        mask_slice = transform(mask_slice.unsqueeze(0))  # Add channel dim

        # Remove batch dim, convert to numpy
        mask_slice = mask_slice.squeeze(0).numpy()
        synthetic_slice = synthetic_tensor[idx]

        # Plot mask
        axes[i, 0].imshow(mask_slice, cmap="gray")
        axes[i, 0].set_title(f"Mask Slice {idx.item()}")
        axes[i, 0].axis("off")

        # Plot synthetic image
        axes[i, 1].imshow(synthetic_slice, cmap="gray")
        axes[i, 1].set_title(f"Synthetic Slice {idx.item()}")
        axes[i, 1].axis("off")

        # Ensure the mask is binary (0s and 1s)
        mask_binary = mask_slice > 0  # Converts any nonzero values to True (1), leaves 0 as False (0)

        # Overlay only the mask regions
        axes[i, 2].imshow(synthetic_slice, cmap="gray")  # Base grayscale image
        axes[i, 2].imshow(mask_binary, cmap="jet", alpha=alpha)  # Apply colormap only to mask regions
        axes[i, 2].set_title(f"Overlay Slice {idx.item()}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def process_directory(directory_path, alpha, vae_model):
    """Find all prostaat.mhd files in the directory, apply elastic transformation, create synthetic image, save, and plot."""

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower() == "prostaat.mhd":
                file_path = os.path.join(root, file)

                # Load original image and mask
                original_image, mask_array = load_mhd_image(file_path)

                # Apply elastic transformation to the mask
                deformed_mask, dx, dy, dz = elastic_transform_3d(mask_array, alpha=alpha)

                # Generate synthetic image using VAE
                with torch.no_grad():
                    mask_tensor = torch.tensor(deformed_mask, dtype=torch.float32).unsqueeze(1).to(
                        device)  # Add channel dim
                    noise = vae.get_noise(mask_tensor.shape[0], z_dim=Z_DIM,
                                          device=device)  # Generate noise for all slices
                    synthetic_image = vae_model.generator(noise, mask_tensor)

                # Convert synthetic image to numpy for saving
                synthetic_image_np = synthetic_image.cpu().numpy().squeeze(1)  # Remove channel dimension

                # Save the deformed mask and synthetic image
                output_deformed_path = os.path.join(root, "prostaat_deformed.mhd")
                save_mhd_image(original_image, deformed_mask, output_deformed_path)
                print(f"Deformed image saved to: {output_deformed_path}")

                output_synthetic_path = os.path.join(root, "mr_bffe_synthetic.mhd")
                save_mhd_image_without_metadata(synthetic_image_np, output_synthetic_path)
                print(f"Synthetic image saved to: {output_synthetic_path}")

                # Plot 5 sample slices
                plot_slices(deformed_mask, synthetic_image_np)


def create_vae_model(checkpoint_path):
    """Load and return a VAE model from a checkpoint file."""
    vae_model = vae.VAE(z_dim=Z_DIM).to(device)
    vae_model.load_state_dict(torch.load(checkpoint_path))
    vae_model.eval()
    return vae_model


if __name__ == "__main__":
    # Load the VAE model
    checkpoint_path = r"C:\Users\20203226\Documents\GitHub\8DM20\code\vae_model_weights_SPADE\vae_model_SPADE.pth"  # Replace with actual path
    vae_model = create_vae_model(checkpoint_path)

    # Process the directory and generate synthetic images
    directory_path = r"D:\capita_selecta\DevelopmentData\DevelopmentData\p102"  # Replace with actual path
    process_directory(directory_path, alpha=1400, vae_model=vae_model)
