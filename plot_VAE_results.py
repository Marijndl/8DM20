import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import vae_SPADE as vae  # VAE model module
from deform_images import load_mhd_image, elastic_transform_3d  # Deformation and loading functions

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VAE model
checkpoint_path = r"./runs/vae_model_weights_SPADE/vae_model_SPADE_6_final.pth"  # Adjust to your checkpoint path
vae_model = vae.VAE(z_dim=256).to(device)
vae_model.load_state_dict(torch.load(checkpoint_path))
vae_model.eval()

# Set patient directory
patient_dir = r"D:\capita_selecta\DevelopmentData\DevelopmentData\p102"  # Adjust to your patient directory

# Load original mask and image
original_mask_path = os.path.join(patient_dir, "prostaat.mhd")
original_image_path = os.path.join(patient_dir, "mr_bffe.mhd")
original_mask_image, original_mask_array = load_mhd_image(original_mask_path)
original_image_image, original_image_array = load_mhd_image(original_image_path)

# Find a representative non-empty slice and an empty slice
non_empty_slices = [i for i in range(original_mask_array.shape[0]) if np.any(original_mask_array[i] > 0)]
empty_slices = [i for i in range(original_mask_array.shape[0]) if not np.any(original_mask_array[i] > 0)]
non_empty_slice_idx = non_empty_slices[len(non_empty_slices) // 2] if non_empty_slices else original_mask_array.shape[
                                                                                                0] // 2
empty_slice_idx = empty_slices[0] if empty_slices else 0  # Take first empty slice or default to 0
print(f"Selected non-empty slice index: {non_empty_slice_idx}")
print(f"Selected empty slice index: {empty_slice_idx}")

# Define alpha and sigma pairs, including empty slice case
cases = [
    ("Empty Slice", None, None, empty_slice_idx),  # No deformation applied, use empty slice
    ("Non-Empty, No Deform.", 0, 50, non_empty_slice_idx),
    ("Non-Empty, Moderate Deform.", 1400, 50, non_empty_slice_idx),
    ("Non-Empty, Large Deform.", 10000, 1000, non_empty_slice_idx)
]

# Create figure with 4 rows and 4 columns
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for row, (label_text, alpha, sigma, slice_idx) in enumerate(cases):
    if alpha is None:  # Empty slice case
        deformed_mask = original_mask_array.copy()  # No deformation, just use original (will be black)
    else:
        deformed_mask, _, _, _ = elastic_transform_3d(original_mask_array, alpha=alpha, sigma=sigma)

    # Generate synthetic image
    with torch.no_grad():
        mask_tensor = torch.tensor(deformed_mask, dtype=torch.float32).unsqueeze(1).to(device)
        mask_tensor += 1  # Shift values as required by the VAE model
        noise = vae.get_noise(mask_tensor.shape[0], z_dim=256, device=device)
        synthetic_image = vae_model.generator(noise, mask_tensor)
        synthetic_image_np = synthetic_image.cpu().numpy().squeeze(1)

    # Plot images for the selected slice
    images = [
        original_mask_array[slice_idx],
        deformed_mask[slice_idx],
        original_image_array[slice_idx],
        synthetic_image_np[slice_idx]
    ]

    for col, img in enumerate(images):
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')

    # Set titles for the first row with larger font
    if row == 0:
        axes[row, 0].set_title("Original Mask", fontsize=25)
        axes[row, 1].set_title("Deformed Mask", fontsize=25)
        axes[row, 2].set_title("Original Image", fontsize=25)
        axes[row, 3].set_title("Synthetic Image", fontsize=25)

    # Set row label with symbols and larger font
    if row == 0:
        label = "Empty Slice"
    else:
        label = f"α={alpha}, σ={sigma}" if alpha != 0 else "α=0 (No Deform.)"
    axes[row, 0].text(-0.1, 0.5, label, transform=axes[row, 0].transAxes, va='center', ha='right', fontsize=25)

plt.suptitle(f"Comparison of the original mask, the deformed mask and the VAE output.", fontsize=30)
plt.tight_layout()
plt.show()