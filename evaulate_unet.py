import random
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import hausdorff_distance

from models import u_net, utils
# to ensure reproducible training/validation split
random.seed(42)

# find out if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path("D:\capita_selecta\DevelopmentData\DevelopmentData")
CHECKPOINTS_DIR = Path.cwd() / "runs" / "segmentation_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "runs/segmentation_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
NO_TEST_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 15
LEARNING_RATE = 1e-4
TOLERANCE = 0.01  # for early stopping

unet_model = u_net.UNet(num_classes=1).to(device)

# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
random.shuffle(patients)

# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS-NO_TEST_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS-NO_TEST_PATIENTS:-NO_TEST_PATIENTS],
    "test": patients[-NO_TEST_PATIENTS:],
}

# Load the best model for evaluation
best_model_path = max(CHECKPOINTS_DIR.glob("u_net-val_loss=*.pth"), key=lambda x: float(x.stem.split('=')[-1]))
unet_model.load_state_dict(torch.load(best_model_path, map_location=device))
unet_model.eval()

# Load test dataset
test_dataset = utils.ProstateMRDataset(partition["test"], IMAGE_SIZE, valid=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

dice_scores = []
hausdorff_distances = []
prediction_3d = []
ground_truth_3d = []

# Test set evaluation
with torch.no_grad():
    for inputs, labels in tqdm(test_dataloader, desc="Evaluating on Test Set"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Model inference using sigmoid activation
        outputs = torch.sigmoid(unet_model(inputs))
        prediction = torch.round(outputs)  # Convert probabilities to binary mask

        # Convert tensors to NumPy arrays
        predicted_masks = prediction.cpu().numpy().astype(int)
        ground_truth_masks = labels.cpu().numpy().astype(int)

        # Compute DICE score only if at least one of the masks contains nonzero pixels
        if np.any(predicted_masks) or np.any(ground_truth_masks):
            intersection = np.sum(predicted_masks * ground_truth_masks)
            union = np.sum(predicted_masks) + np.sum(ground_truth_masks)
            dice_score = (2. * intersection) / (union + 1e-6)  # Avoid division by zero
            dice_scores.append(dice_score)

        # Compute Hausdorff distance only if both masks contain nonzero pixels
        if np.any(predicted_masks) and np.any(ground_truth_masks):
            hausdorff_dist = hausdorff_distance(predicted_masks[0, 0], ground_truth_masks[0, 0])
            hausdorff_distances.append(hausdorff_dist)
            
        prediction_3d.append(predicted_masks)
        ground_truth_3d.append(ground_truth_masks)

    prediction_3d = np.array(prediction_3d)
    ground_truth_3d = np.array(ground_truth_3d)
    predictions_3d = np.array([prediction_3d[:86], prediction_3d[86:]])
    ground_truths_3d = np.array([ground_truth_3d[:86], ground_truth_3d[86:]])

    dice_scores_3d = []
    for i in range(2):
        intersection_3d = np.sum(predictions_3d[i] * ground_truths_3d[i])
        union_3d = np.sum(predictions_3d[i]) + np.sum(ground_truths_3d[i])
        dice_score_3d = (2.0 * intersection_3d) / (union_3d + 1e-6)  # Avoid division by zero
        dice_scores_3d.append(dice_score_3d)

# Convert lists to NumPy arrays
dice_scores = np.array(dice_scores) if dice_scores else np.array([np.nan])  # Handle empty case
hausdorff_distances = np.array(hausdorff_distances) if hausdorff_distances else np.array([np.nan])  # Handle empty case

print(repr(dice_scores.tolist()))
print(repr(hausdorff_distances.tolist()))

# Print final metrics
print(f"Test Set Evaluation:")
if not np.isnan(dice_scores).all():
    print(f" - Mean DICE Score: {dice_scores.mean():.4f} ± {dice_scores.std():.4f}")
else:
    print(" - DICE Score: Not computed (all masks were empty)")

if not np.isnan(hausdorff_distances).all():
    print(f" - Mean Hausdorff Distance: {hausdorff_distances.mean():.4f} ± {hausdorff_distances.std():.4f}")
else:
    print(" - Hausdorff Distance: Not computed (at least one mask was empty for all cases)")

print(f" - 3D DICE scores : {tuple(dice_scores_3d)}")
