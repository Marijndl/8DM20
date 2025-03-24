import os
from deform_images import load_mhd_image, elastic_transform_3d, save_mhd_image

def process_directory(directory_path, alpha):
    """Find all prostaat.mhd files in the directory, apply elastic transformation, and save the deformed images."""
    # List all files in the directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower() == "prostaat.mhd":
                file_path = os.path.join(root, file)

                # Load, apply deformation, and save
                original_image, mask_array = load_mhd_image(file_path)

                deformed_mask, dx, dy, dz = elastic_transform_3d(mask_array, alpha=alpha)  # Now returns displacement fields

                # Save the deformed volume
                output_path = os.path.join(root, "prostaat_deformed.mhd")
                save_mhd_image(original_image, deformed_mask, output_path)
                print(f"Deformed image saved to: {output_path}")


if __name__ == "__main__":
    # Deform the images
    directory_path = r"D:\capita_selecta\DevelopmentData\DevelopmentData"  # Replace with actual path
    process_directory(directory_path, alpha=1400)
