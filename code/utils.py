import torch
import numpy as np
import SimpleITK as sitk
import random
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ProstateMRDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.

    Parameters
    ----------
    paths : list[Path]
        Paths to the patient data.
    img_size : list[int]
        Size of images to be interpolated to.
    valid : bool, optional
        Whether the dataset is used for validation (default: False).
    synthetic : bool, optional
        Whether synthetic images should be included (default: False).
    """

    def __init__(self, paths, img_size, valid=False, synthetic=False):
        self.mr_image_list = []
        self.mask_list = []
        self.mr_image_list_synthetic = []
        self.mask_list_synthetic = []
        self.synthetic = synthetic

        # Load images
        for path in paths:
            self.mr_image_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe.mhd")).astype(np.int32)
            )
            self.mask_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd")).astype(np.int32)
            )
            if synthetic:
                self.mr_image_list_synthetic.append(
                    sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe_synthetic.mhd")).astype(np.int32)
                )
                self.mask_list_synthetic.append(
                    sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat_deformed.mhd")).astype(np.int32)
                )

        # Compute dataset length: If synthetic is included, we double the slice count
        self.no_patients = len(self.mr_image_list)
        self.no_slices = self.mr_image_list[0].shape[0]
        self.total_slices = self.no_patients * self.no_slices
        if synthetic:
            self.total_slices *= 2  # Since we now include both real and synthetic slices

        # Compute normalization parameters
        self.train_data_mean = np.mean(self.mr_image_list)
        self.train_data_std = np.std(self.mr_image_list)
        self.norm_transform = transforms.Normalize(self.train_data_mean, self.train_data_std)

        if synthetic:
            self.synth_data_mean = np.mean(self.mr_image_list_synthetic)
            self.synth_data_std = np.std(self.mr_image_list_synthetic)
            self.norm_transform_synth = transforms.Normalize(self.synth_data_mean, self.synth_data_std)

        # Define transformations for normal images
        base_transforms = [
            transforms.ToPILImage(mode="I"),
            transforms.CenterCrop(256),
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]
        if not valid:
            base_transforms.append(
                transforms.RandomAffine(
                    degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.9, 1.1)
                )
            )
        self.img_transform = transforms.Compose(base_transforms)

        # Define transformations for synthetic images
        if synthetic:
            synth_base_transforms = [
                transforms.ToPILImage(mode="I"),
                transforms.CenterCrop(256),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
            if not valid:
                synth_base_transforms.append(
                    transforms.RandomAffine(
                        degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.95, 1.05)
                    )
                )
            self.synthetic_transform = transforms.Compose(synth_base_transforms)

    def __len__(self):
        """Returns length of dataset, considering both real and synthetic images if applicable."""
        if self.synthetic:
            # Double the dataset size to account for both real and synthetic slices
            return self.no_patients * self.no_slices * 2
        else:
            return self.no_patients * self.no_slices

    def __getitem__(self, index):
        """Returns a randomly chosen real or synthetic MR image slice along with its corresponding mask."""

        # Determine if we are using real or synthetic data
        use_synthetic = self.synthetic and (random.random() < 0.5)  # 50% chance to use synthetic data

        if use_synthetic:
            # Get synthetic slice
            adjusted_index = index % (self.no_patients * self.no_slices)
            patient = adjusted_index // self.no_slices
            the_slice = adjusted_index % self.no_slices
            img_list = self.mr_image_list_synthetic
            mask_list = self.mask_list_synthetic
            transform = self.synthetic_transform
            norm_transform = self.norm_transform_synth
        else:
            # Get real slice
            patient = index // self.no_slices
            the_slice = index % self.no_slices
            img_list = self.mr_image_list
            mask_list = self.mask_list
            transform = self.img_transform
            norm_transform = self.norm_transform

        # Seed randomness to ensure consistency in image-mask transformation
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)

        x = norm_transform(transform(img_list[patient][the_slice, ...]).float())

        random.seed(seed)
        torch.manual_seed(seed)

        y = transform((mask_list[patient][the_slice, ...] > 0).astype(np.int32))

        return x, y
    
class DiceBCELoss(nn.Module):
    """Loss function, computed as the sum of Dice score and binary cross-entropy.

    Notes
    -----
    This loss assumes that the inputs are logits (i.e., the outputs of a linear layer),
    and that the targets are integer values that represent the correct class labels.
    """

    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, outputs, targets, smooth=1):
        """Calculates segmentation loss for training

        Parameters
        ----------
        outputs : torch.Tensor
            predictions of segmentation model
        targets : torch.Tensor
            ground-truth labels
        smooth : float
            smooth parameter for dice score avoids division by zero, by default 1

        Returns
        -------
        float
            the sum of the dice loss and binary cross-entropy
        """
        outputs = torch.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # compute Dice
        intersection = (outputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            outputs.sum() + targets.sum() + smooth
        )
        BCE = nn.functional.binary_cross_entropy(outputs, targets, reduction="mean")

        return BCE + dice_loss

