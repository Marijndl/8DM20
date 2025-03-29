# Prostate MRI Analysis with Deep Learning
This repository contains code for prostate segmentation in MRI images using a standard U-Net. To improve segmentation performance, the repository includes code for generating synthetic data with a Variational Autoencoder (VAE), both with and without SPADE layers, enabling conditional generation.

This work was conducted for the course 8DM20: Capita Selecta for Medical Imaging in the Department of Biomedical Engineering at Eindhoven University of Technology.
## Table of Contents

1.  [Contents](#contents)
2.  [Setup](#setup)
    *   [Clone the Repository](#clone-the-repository)
    *   [Install Dependencies](#install-dependencies)
    *   [Data Preparation](#data-preparation)
3.  [Usage](#usage)
    *   [Training](#training)
        *   [U-Net Segmentation](#u-net-segmentation)
        *   [VAE](#vae)
        *   [VAE with SPADE](#vae-with-spade)
    *   [Evaluation](#evaluation)
        *   [U-Net Evaluation](#u-net-evaluation)
    *   [Inference](#inference)
        *   [Applying Segmentation](#applying-segmentation)
        *   [Generating Images with VAE](#generating-images-with-vae)
    *   [Data Augmentation and Synthetic Image Generation](#data-augmentation-and-synthetic-image-generation)
        * [Deforming Images and Generating Synthetic Data](#deforming-images-and-generating-synthetic-data-1)
4.  [Configuration](#configuration)
5.  [Key Parameters and Tuning](#key-parameters-and-tuning)
6.  [Evaluation Metrics](#evaluation-metrics)

## Authors

Willem Pladet* (1606492)\
Joris Mentink* (1614568)\
Joost Klis* (1503715)\
Marijn de Lange* (1584944)\
Bruno Rütten* (1579320)\
Noach Schilt* (1584979)

*: Technical University Eindhoven, Eindhoven, The Netherlands

## Contents

*   **`models/`**: Contains the definitions of the neural network models.
    *   `u_net.py`: U-Net architecture for prostate segmentation.
    *   `vae.py`: Standard Variational Autoencoder (VAE) for image generation.
    *   `vae_SPADE.py`: VAE with SPADE layers for conditional image generation based on segmentation maps.
    *   `utils.py`: Utility functions, including data loading (`ProstateMRDataset`), loss functions (`DiceBCELoss`), and data transformations.
*   **Scripts:**
    *   `train_unet.py`: Script for training the U-Net segmentation model.
    *   `train_vae.py`: Script for training the standard VAE.
    *   `train_vae_SPADE.py`: Script for training the SPADE-enhanced VAE.
    *   `apply_segmentation.py`: Applies a trained U-Net model to segment a sample MRI image.
    *   `apply_vae.py`: Generates images from a trained VAE model using random noise.
    *   `deform_images.py`:  Performs elastic deformation on images (for data augmentation).  Includes visualization tools to inspect the deformation.
    *   `deform_all_images.py`: Applies elastic deformation to prostate masks and uses the `vae_SPADE` model to generate corresponding synthetic MR images from deformed masks. This script helps in creating synthetic MR data with different appearances based on mask deformation.
    *   `evaulate_unet.py`: Evaluates a trained U-Net model using Dice score and Hausdorff distance.
*   **`requirements.txt`**: List of Python package dependencies.
*   **`.gitignore`**: Specifies files and directories to be ignored by Git.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Marijndl/8DM20
    cd 8DM20
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Preparation:**

    *   The code expects the data to be organized in a directory structure like this:

        ```
        DevelopmentData/
        ├── p101/
        │   ├── mr_bffe.mhd
        │   ├── prostaat.mhd
        │   ├── mr_bffe_synthetic.mhd    (optional, created by deform_all_images.py)
        │   └── prostaat_deformed.mhd     (optional, created by deform_all_images.py)
        ├── p102/
        │   ├── mr_bffe.mhd
        │   ├── prostaat.mhd
        │   └── ...
        └── ...
        ```

    *   Each patient (`p101`, `p102`, etc.) should have a directory containing:
        *   `mr_bffe.mhd`: The MRI image.
        *   `prostaat.mhd`: The prostate segmentation mask.
        *   The synthetic images and deformed prostates are optional and will be created when using `deform_all_images.py`.
    *   Update the `DATA_DIR` variable in the training and evaluation scripts to point to the location of your data.

## Usage

### Training

1.  **U-Net Segmentation:**

    ```bash
    python train_unet.py
    ```

    *   Training checkpoints will be saved in the `runs/segmentation_model_weights/` directory.  TensorBoard logs will be in the `runs/segmentation_runs/` directory.

2.  **VAE:**

    ```bash
    python train_vae.py
    ```

    *   Training checkpoints will be saved in the `runs/vae_model_weights/` directory.  TensorBoard logs will be in the `runs/vae_runs/` directory.

3.  **VAE with SPADE:**

    ```bash
    python train_vae_SPADE.py
    ```

    *   Training checkpoints will be saved in the `runs/vae_model_weights_SPADE/` directory.  TensorBoard logs will be in the `runs/vae_runs_SPADE/` directory.

### Evaluation

1.  **U-Net Evaluation:**

    ```bash
    python evaulate_unet.py
    ```

    *   This script will load the best U-Net model (based on validation loss) and evaluate it on the test set, reporting the mean Dice score and Hausdorff distance.

### Inference

1.  **Applying Segmentation:**

    ```bash
    python apply_segmentation.py
    ```

    *   This script loads a trained U-Net model and applies it to a sample image from the validation set, displaying the input image, ground truth mask, and predicted segmentation.

2.  **Generating Images with VAE:**

    ```bash
    python apply_vae.py
    ```

    *   This script loads a trained VAE model and generates a sample image from random noise.

### Data Augmentation and Synthetic Image Generation

1.  **Deforming Images and Generating Synthetic Data:**

    ```bash
    python deform_all_images.py
    ```

    *   This script applies elastic deformation to prostate masks located in the `DevelopmentData` directory (or the directory you specify).
    *   For each deformed mask, it generates a synthetic MRI image using the trained `vae_SPADE` model.
    *   The deformed masks and synthetic images are saved in the same directory as the original data.
    *   To fine-tune the degree of deformation, run `deform_images.py` to get a better understanding through some visualisations. 

## Configuration

Most hyperparameters (learning rate, batch size, number of epochs, etc.) are currently hardcoded in the training scripts. To modify these, edit the corresponding Python files directly.

## Key Parameters and Tuning

*   **`train_unet.py`**:
    *   `LEARNING_RATE`:  Learning rate for the U-Net optimizer.
    *   `BATCH_SIZE`:  Batch size during training.
    *   `N_EPOCHS`: Number of training epochs.
    *   `TOLERANCE`: Early stopping tolerance.

*   **`train_vae.py` and `train_vae_SPADE.py`**:
    *   `LEARNING_RATE`: Learning rate for the VAE optimizer.
    *   `BATCH_SIZE`: Batch size during training.
    *   `N_EPOCHS`: Number of training epochs.
    *   `Z_DIM`: Dimension of the VAE's latent space.
    *   `DECAY_LR_AFTER`: The epoch after which the learning rate will decay.
*   **`deform_images.py`**:
    *    `alpha`: Controls the intensity of deformation
    *   `sigma`: Controls the smoothness of the deformation

## Evaluation Metrics

*   **Dice Score**: A measure of overlap between the predicted segmentation and the ground truth.  A higher Dice score indicates better segmentation performance.
*   **Hausdorff Distance**: Measures the distance between the boundaries of the predicted segmentation and the ground truth. A lower Hausdorff distance indicates better segmentation accuracy.
