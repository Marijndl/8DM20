import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torchvision.utils import make_grid
from tqdm import tqdm

import utils
import vae_SPADE as vae

# to ensure reproducible training/validation split
random.seed(41)

# find out if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# directories with data and to store training checkpoints and logs
DATA_DIR = Path("D:\capita_selecta\DevelopmentData\DevelopmentData")
CHECKPOINTS_DIR = Path.cwd() / "vae_model_weights_SPADE"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "vae_runs_SPADE"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 100
DECAY_LR_AFTER = 50
LEARNING_RATE = 1e-4
DISPLAY_FREQ = 10
TOLERANCE = 0.03  # for early stopping

# dimension of VAE latent space
Z_DIM = 256

# # function to reduce the
# def lr_lambda(the_epoch):
#     """Function for scheduling learning rate"""
#     return (
#         1.0
#         if the_epoch < DECAY_LR_AFTER
#         else 1 - float(the_epoch - DECAY_LR_AFTER) / (N_EPOCHS - DECAY_LR_AFTER)
#     )

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
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load training data and create DataLoader with batching and shuffling
dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE, valid=True, synthetic=False)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE, valid=True, synthetic=False)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# initialise model, optimiser
vae_model = vae.VAE(z_dim=Z_DIM).to(device)
optimizer = torch.optim.Adam(vae_model.parameters(), lr=LEARNING_RATE)

# Compute total number of training steps
total_steps = N_EPOCHS * len(dataloader)  # total training steps
warmup_steps = 10 * len(dataloader)  # warmup lasts for 10 epochs worth of steps

# Define warmup function based on batch step
def lr_lambda(current_step):
    if current_step < warmup_steps:
        return (current_step + 1) / warmup_steps  # Linear warmup
    else:
        return 1  # After warmup, let CosineAnnealing take over

warmup_scheduler = LambdaLR(optimizer, lr_lambda)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - 4 * warmup_steps, eta_min=1e-6)

# training loop
minimum_valid_loss = 10  # initial validation loss
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary
for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0
    
    # Training iterations with tqdm showing epoch number
    train_loader = tqdm(dataloader, position=0, leave=True)
    train_loader.set_description(f"Epoch {epoch+1}/{N_EPOCHS} [Training]")

    for x_real, y_real in train_loader:
    # Converteer naar float
        x_real, y_real = x_real.to(device).float(), y_real.to(device).float()

        # Adjust the label so black also contains information
        y_real = y_real + 1

        optimizer.zero_grad()
        x_recon, mu, logvar = vae_model(x_real, y_real)

        loss = vae.vae_loss(x_real, x_recon, mu, logvar)
        loss.backward()
        optimizer.step()

        current_train_loss += loss.item()

        # Step the scheduler
        if epoch < 10:
            warmup_scheduler.step()
        elif epoch < 40:
            pass
        else:
            cosine_scheduler.step()


    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)

    # evaluate validation loss
    with torch.no_grad():
        vae_model.eval()
        valid_loader = tqdm(valid_dataloader, position=0, leave=True)
        valid_loader.set_description(f"Epoch {epoch+1}/{N_EPOCHS} [Validation]")

        for x_real, y_real in valid_loader:
            x_real, y_real = x_real.to(device).float(), y_real.to(device).float()

            # Adjust the label so black also contains information
            y_real = y_real + 1

            x_recon, mu, logvar = vae_model(x_real, y_real)
            # Gebruik een lagere beta om posterior collapse te voorkomen
            loss = vae.vae_loss(x_real, x_recon, mu, logvar, beta=0.1)
            current_valid_loss += loss.item()

        # write to tensorboard log
        writer.add_scalar(
            "Loss/validation", current_valid_loss / len(valid_dataloader), epoch
        )

        # save examples of real/fake images
        if (epoch + 1) % DISPLAY_FREQ == 0:
            x_recon = x_recon.detach().cpu()
            x_real = x_real.detach().cpu()

            img_grid = make_grid(
                torch.cat((x_recon[:5], x_real[:5])), nrow=5, padding=12, pad_value=-1
            )
            writer.add_image(
                "Real_fake", np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5, epoch + 1
            )

            # Gebruik y_real in plaats van random_segmap
            noise = vae.get_noise(10, z_dim=Z_DIM, device=device)
            image_samples = vae_model.generator(noise, y_real[:10])  # Use some random masks
            img_grid = make_grid(
                torch.cat((image_samples[:5].cpu(), image_samples[5:].cpu())),
                nrow=5,
                padding=12,
                pad_value=-1,
            )
            writer.add_image(
                "Samples",
                np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5,
                epoch + 1,
            )
        vae_model.train()

    # if validation loss is improving, save model checkpoint
    # only start saving after 5 epochs
    if (current_valid_loss / len(valid_dataloader)) < minimum_valid_loss + TOLERANCE:
        minimum_valid_loss = current_valid_loss / len(valid_dataloader)
        weights_dict = {k: v.cpu() for k, v in vae_model.state_dict().items()}
        if epoch > 5:
            torch.save(
                weights_dict,
                CHECKPOINTS_DIR / f"vae_model_SPADE_6.pth",

weights_dict = {k: v.cpu() for k, v in vae_model.state_dict().items()}
torch.save(
    weights_dict,
    CHECKPOINTS_DIR / "vae_model_SPADE_6_final.pth",
)