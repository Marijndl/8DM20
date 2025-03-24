import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter, zoom
from mpl_toolkits.mplot3d import Axes3D


def load_mhd_image(file_path):
    """Load a 3D binary mask from an .mhd and .zraw file."""
    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)  # Shape: (depth, height, width)
    return image, array


def elastic_transform_3d(image, alpha=1400, sigma=50):
    """Apply large-scale 3D elastic deformation to the whole volume while preserving binary values."""
    shape = image.shape

    # Define low-resolution grid for coarse displacement
    low_res_factor = 4  # Downsampling factor (adjustable)
    low_res_shape = (shape[0] // low_res_factor, shape[1] // low_res_factor, shape[2] // low_res_factor)

    # Generate smooth displacement fields at low resolution
    dx = gaussian_filter((np.random.rand(*low_res_shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*low_res_shape) * 2 - 1), sigma) * alpha
    dz = gaussian_filter((np.random.rand(*low_res_shape) * 2 - 1), sigma) * alpha

    # Upsample displacement fields to match the original image shape (using nearest interpolation)
    dx = zoom(dx, np.array(shape) / np.array(dx.shape), order=0)
    dy = zoom(dy, np.array(shape) / np.array(dy.shape), order=0)
    dz = zoom(dz, np.array(shape) / np.array(dz.shape), order=0)

    # Ensure the displacement fields have the exact same shape as the input image
    dx, dy, dz = dx[:shape[0], :shape[1], :shape[2]], dy[:shape[0], :shape[1], :shape[2]], dz[:shape[0], :shape[1],
                                                                                           :shape[2]]

    # Meshgrid for pixel indices
    z, y, x = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )

    # Flatten the arrays for interpolation
    indices = np.vstack([(z + dz).ravel(), (y + dy).ravel(), (x + dx).ravel()])

    # Apply elastic deformation with nearest-neighbor interpolation
    deformed_image = map_coordinates(image, indices, order=0, mode="reflect").reshape(shape)

    # Ensure binary output (set values below 0.5 to 0 and above to 1)
    deformed_image = (deformed_image > 0.5).astype(np.uint8)

    return deformed_image, dx, dy, dz


def save_mhd_image(original_image, array, output_path):
    """Save the modified 3D mask as an .mhd and .zraw file."""
    new_image = sitk.GetImageFromArray(array)
    new_image.CopyInformation(original_image)  # Preserve metadata
    sitk.WriteImage(new_image, output_path)

def save_mhd_image_without_metadata(array, output_path, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    """
    Save a 3D numpy array as an .mhd and .zraw file without copying metadata.
    """
    new_image = sitk.GetImageFromArray(array)

    # Assign basic metadata (default values can be modified if needed)
    new_image.SetSpacing(spacing)
    new_image.SetOrigin(origin)
    new_image.SetDirection(direction)

    sitk.WriteImage(new_image, output_path)
    print(f"Image saved without metadata at: {output_path}")


def plot_examples(original, deformed, num_examples=3):
    """Plot original, deformed, and difference slices, ensuring foreground pixels are present."""

    # Find slices that contain white pixels (foreground)
    non_empty_slices = [i for i in range(original.shape[0]) if np.any(original[i] > 0)]
    print("Number of non-empty slices: {}".format(len(non_empty_slices)))
    false_count = np.sum(deformed == 0)
    true_count = np.sum(deformed == 1)
    total_pixels = deformed.size

    print(f"False pixels: {false_count}")
    print(f"True pixels: {true_count}")
    print(f"True + false pixels: {false_count + true_count}")
    print(f"Total pixels: {total_pixels}")

    if len(non_empty_slices) == 0:
        print("No valid slices found with white pixels. Skipping plotting.")
        return

    # Select random slices from those that contain foreground pixels
    num_to_plot = min(num_examples, len(non_empty_slices))
    indices = np.random.choice(non_empty_slices, num_to_plot, replace=False)

    fig, axes = plt.subplots(num_to_plot, 3, figsize=(12, 4 * num_to_plot))

    for i, idx in enumerate(indices):
        difference = np.abs(original[idx] - deformed[idx])  # Absolute difference map

        axes[i, 0].imshow(original[idx], cmap="gray")
        axes[i, 0].set_title(f"Original Slice {idx}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(deformed[idx], cmap="gray")
        axes[i, 1].set_title(f"Deformed Slice {idx}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(difference, cmap="hot")  # Highlight differences
        axes[i, 2].set_title(f"Difference Slice {idx}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_deformation_field_3d(dx, dy, dz, mask_array, spacing=8):
    """
    Plot the 3D deformation field with arrows colored by direction.

    Args:
        dx, dy, dz: Displacement fields in x, y, z directions
        mask_array: Original mask to determine region of interest
        spacing: Sampling spacing for arrows (to avoid overcrowding)
    """
    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Find the region of interest (where the mask has values)
    z_indices, y_indices, x_indices = np.where(mask_array > 0)

    # If the mask is empty or very small, sample from the entire volume
    if len(z_indices) < 100:
        z_indices, y_indices, x_indices = np.where(np.ones_like(mask_array) > 0)

    # Sample points with spacing to avoid overcrowding
    sample_indices = np.arange(0, len(z_indices), spacing)
    z_sample = z_indices[sample_indices]
    y_sample = y_indices[sample_indices]
    x_sample = x_indices[sample_indices]

    # Get displacement vectors at sampled points
    u = dx[z_sample, y_sample, x_sample]  # x-displacement
    v = dy[z_sample, y_sample, x_sample]  # y-displacement
    w = dz[z_sample, y_sample, x_sample]  # z-displacement

    # Calculate the magnitude of displacement for coloring
    magnitude = np.sqrt(u ** 2 + v ** 2 + w ** 2)

    # Normalize displacements for better visualization
    scale_factor = 5.0 / np.max(magnitude) if np.max(magnitude) > 0 else 1.0
    u = u * scale_factor
    v = v * scale_factor
    w = w * scale_factor

    # Create a colormap based on direction
    # Using RGB: (R,G,B) = normalized (x,y,z) displacement
    colors = np.zeros((len(u), 4))  # RGBA

    # Normalize for coloring (-1 to 1 range for each dimension)
    max_disp = np.max([np.abs(u), np.abs(v), np.abs(w)])
    if max_disp > 0:
        norm_u = u / max_disp
        norm_v = v / max_disp
        norm_w = w / max_disp

        # R for x displacement (positive = red, negative = cyan)
        colors[:, 0] = 0.5 + 0.5 * norm_u
        # G for y displacement (positive = green, negative = magenta)
        colors[:, 1] = 0.5 + 0.5 * norm_v
        # B for z displacement (positive = blue, negative = yellow)
        colors[:, 2] = 0.5 + 0.5 * norm_w
        # Alpha (transparency)
        colors[:, 3] = 0.8  # Fixed transparency

    # Plot arrows representing the deformation field
    ax.quiver(x_sample, y_sample, z_sample, u, v, w, colors=colors, length=1.0)

    # Add a colorbar legend to explain the color mapping
    # We'll create a custom colorbar
    ax_cb = fig.add_axes([0.88, 0.1, 0.03, 0.8])  # [left, bottom, width, height]

    # Create a gradient for the colorbar - FIXED BROADCASTING ISSUE
    gradient = np.linspace(-1, 1, 256)

    # Create separate gradient arrays for each direction
    x_gradient = np.zeros((256, 3))
    y_gradient = np.zeros((256, 3))
    z_gradient = np.zeros((256, 3))

    # X direction (Red channel)
    x_gradient[:, 0] = 0.5 + 0.5 * gradient
    x_gradient[:, 1] = 0.5
    x_gradient[:, 2] = 0.5

    # Y direction (Green channel)
    y_gradient[:, 0] = 0.5
    y_gradient[:, 1] = 0.5 + 0.5 * gradient
    y_gradient[:, 2] = 0.5

    # Z direction (Blue channel)
    z_gradient[:, 0] = 0.5
    z_gradient[:, 1] = 0.5
    z_gradient[:, 2] = 0.5 + 0.5 * gradient

    # Create stacked gradient image
    gradient_img = np.vstack([x_gradient, y_gradient, z_gradient])

    # Plot gradient
    ax_cb.imshow(gradient_img, aspect='auto', extent=[0, 1, -1, 1])
    ax_cb.set_yticks([-1, -0.33, 0.33, 1])
    ax_cb.set_yticklabels(['-Z', '-Y', '+Y', '+Z'])

    # Add labels to the colorbar
    ax_cb.text(1.5, 0.67, 'X (Red)', rotation=90, va='center')
    ax_cb.text(1.5, 0, 'Y (Green)', rotation=90, va='center')
    ax_cb.text(1.5, -0.67, 'Z (Blue)', rotation=90, va='center')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Deformation Field Visualization')

    # Set axis limits to match the mask dimensions
    ax.set_xlim(0, mask_array.shape[2])
    ax.set_ylim(0, mask_array.shape[1])
    ax.set_zlim(0, mask_array.shape[0])

    plt.tight_layout()
    plt.show()


def plot_deformation_field_slices(dx, dy, dz, mask_array, num_slices=3):
    """
    Plot 2D slices of the deformation field with arrows colored by direction.

    Args:
        dx, dy, dz: Displacement fields in x, y, z directions
        mask_array: Original mask to determine region of interest
        num_slices: Number of slices to plot
    """
    # Find slices that contain foreground pixels
    non_empty_slices = [i for i in range(mask_array.shape[0]) if np.any(mask_array[i] > 0)]

    if len(non_empty_slices) == 0:
        print("No valid slices found with foreground pixels. Using middle slices.")
        non_empty_slices = [mask_array.shape[0] // 4, mask_array.shape[0] // 2, 3 * mask_array.shape[0] // 4]

    # Select slices to plot
    num_to_plot = min(num_slices, len(non_empty_slices))
    slice_indices = np.random.choice(non_empty_slices, num_to_plot, replace=False)
    slice_indices.sort()  # Sort for better visualization

    fig, axes = plt.subplots(1, num_to_plot, figsize=(6 * num_to_plot, 6))
    if num_to_plot == 1:
        axes = [axes]  # Make axes iterable when there's only one subplot

    for i, slice_idx in enumerate(slice_indices):
        # Get the 2D displacement field for this slice
        dx_slice = dx[slice_idx]
        dy_slice = dy[slice_idx]
        mask_slice = mask_array[slice_idx]

        # Downsample for clearer visualization (adjust spacing as needed)
        spacing = max(mask_slice.shape[0] // 20, mask_slice.shape[1] // 20, 1)
        y_grid, x_grid = np.mgrid[0:mask_slice.shape[0]:spacing, 0:mask_slice.shape[1]:spacing]

        # Sample the displacement fields
        u = dx_slice[y_grid, x_grid]  # x-displacement
        v = dy_slice[y_grid, x_grid]  # y-displacement

        # Calculate magnitude for scaling
        magnitude = np.sqrt(u ** 2 + v ** 2)
        max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0

        # Normalize for better visualization
        scale = 0.5 * spacing / max_mag if max_mag > 0 else 1.0
        u = u * scale
        v = v * scale

        # Create a colormap based on direction
        # Hue represents direction, Value/Saturation represents magnitude
        angles = np.arctan2(v, u) * 180 / np.pi  # Convert to degrees
        normalized_angles = (angles + 180) / 360  # Normalize to [0, 1]

        # Display the background image (mask)
        axes[i].imshow(mask_slice, cmap='gray', alpha=0.5)

        # Plot the quiver with HSV coloring based on direction
        quiver = axes[i].quiver(x_grid, y_grid, u, v, normalized_angles,
                                cmap='hsv', scale=1, scale_units='xy',
                                angles='xy', width=0.002, headwidth=4)

        # Add a colorbar
        cbar = plt.colorbar(quiver, ax=axes[i])
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['180°', '270°', '0°', '90°', '180°'])
        cbar.set_label('Direction')

        axes[i].set_title(f'Deformation Field - Slice {slice_idx}')
        axes[i].set_xlim(0, mask_slice.shape[1])
        axes[i].set_ylim(mask_slice.shape[0], 0)  # Invert y-axis for image coordinates
        axes[i].set_aspect('equal')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example Usage
    file_path = r"D:\capita_selecta\DevelopmentData\DevelopmentData\p102\prostaat.mhd"  # Replace with actual path
    output_path = r"D:\capita_selecta\DevelopmentData\DevelopmentData\p102\prostaat_deformed.mhd"

    # Load, apply deformation, and save
    original_image, mask_array = load_mhd_image(file_path)
    deformed_mask, dx, dy, dz = elastic_transform_3d(mask_array)  # Now returns displacement fields

    # Save the deformed volume (uncomment to enable saving)
    # save_mhd_image(original_image, deformed_mask, output_path)

    # Plot some examples of original vs deformed slices
    plot_examples(mask_array, deformed_mask, num_examples=3)

    # Plot the 3D deformation field
    plot_deformation_field_3d(dx, dy, dz, mask_array, spacing=15)

    # Plot 2D slices of the deformation field
    plot_deformation_field_slices(dx, dy, dz, mask_array, num_slices=3)