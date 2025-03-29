import random
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage.metrics import hausdorff_distance

import u_net
import utils

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
CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "segmentation_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
NO_TEST_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 100
LEARNING_RATE = 1e-4
TOLERANCE = 0.01  # for early stopping

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

# load training data and create DataLoader with batching and shuffling
dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE, synthetic=True)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE, valid=True, synthetic=True)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# initialise model, optimiser, and loss function
loss_function = utils.DiceBCELoss()
unet_model = u_net.UNet(num_classes=1).to(device)
optimizer = torch.optim.Adam(unet_model.parameters(), lr=LEARNING_RATE)

minimum_valid_loss = 10  # initial validation loss
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary

# training loop
for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0

    # Training iterations with tqdm showing epoch number
    train_loader = tqdm(dataloader, position=0, leave=True)
    train_loader.set_description(f"Epoch {epoch+1}/{N_EPOCHS} [Training]")

    for inputs, labels in train_loader:
        # needed to zero gradients in each iterations
        optimizer.zero_grad()
        outputs = unet_model(inputs.to(device))  # forward pass
        loss = loss_function(outputs, labels.to(device).float())
        loss.backward()  # backpropagate loss
        current_train_loss += loss.item()
        optimizer.step()  # update weights

    # evaluate validation loss
    with torch.no_grad():
        unet_model.eval()

        valid_loader = tqdm(valid_dataloader, position=0, leave=True)
        valid_loader.set_description(f"Epoch {epoch+1}/{N_EPOCHS} [Validation]")

        for inputs, labels in valid_loader:
            outputs = unet_model(inputs.to(device))  # forward pass
            loss = loss_function(outputs, labels.to(device).float())
            current_valid_loss += loss.item()

        unet_model.train()

    # write to tensorboard log
    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)
    writer.add_scalar(
        "Loss/validation", current_valid_loss / len(valid_dataloader), epoch
    )

    # if validation loss is improving, save model checkpoint
    # only start saving after 10 epochs
    if (current_valid_loss / len(valid_dataloader)) < minimum_valid_loss + TOLERANCE:
        minimum_valid_loss = current_valid_loss / len(valid_dataloader)
        weights_dict = {k: v.cpu() for k, v in unet_model.state_dict().items()}
        if epoch > 5:
            torch.save(
                weights_dict,
                CHECKPOINTS_DIR / f"u_net-val_loss={minimum_valid_loss:04}.pth",
            )

########################################################################
# Test set evaluation

# Load the best model for evaluation
best_model_path = max(CHECKPOINTS_DIR.glob("u_net-val_loss=*.pth"), key=lambda x: float(x.stem.split('=')[-1]))
unet_model.load_state_dict(torch.load(best_model_path, map_location=device))
unet_model.eval()

# Load test dataset
test_dataset = utils.ProstateMRDataset(partition["test"], IMAGE_SIZE, valid=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

dice_scores = []
hausdorff_distances = []

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

# Convert lists to NumPy arrays
dice_scores = np.array(dice_scores) if dice_scores else np.array([np.nan])  # Handle empty case
hausdorff_distances = np.array(hausdorff_distances) if hausdorff_distances else np.array([np.nan])  # Handle empty case

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
