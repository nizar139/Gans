import os
import itertools

from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, limit=1000):
        """
        Args:
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.transform = transform

        # List all files in the directory
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))][:limit]

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Retrieves an image sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (Tensor): The transformed image.
        """
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')  # Open image and convert to RGB

        if self.transform:
            image = self.transform(image)

        return image

    def get_infinite_iterator(self, batch_size, num_workers=0):
        """
        Creates an infinite iterator that yields batches of data.

        Args:
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.

        Returns:
            Iterator that yields batches of data.
        """
        data_loader = DataLoader(self, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        # Create an infinite iterator over the DataLoader
        return itertools.cycle(data_loader)

    

def generate_CutMix_samples(real_batch, fake_batch, D_unet, device=torch.device('cpu')):
    batch_size, _, H, W = real_batch.size()

    # Generate random ratios for the batch
    ratios = torch.rand(batch_size, device=device)

    # Randomly permute the fake batch for CutMix
    rand_indices = torch.randperm(batch_size, device=device)

    target_a = real_batch
    target_b = fake_batch[rand_indices]

    # Generate bounding boxes for each image in the batch
    bbx1, bby1, bbx2, bby2 = rand_bbox(real_batch.size(), ratios, device)

    # Create masks for CutMix
    x = torch.arange(W, device=device).unsqueeze(0)
    y = torch.arange(H, device=device).unsqueeze(0)
    mask_x = (x >= bbx1.view(-1, 1)) & (x < bbx2.view(-1, 1))
    mask_y = (y >= bby1.view(-1, 1)) & (y < bby2.view(-1, 1))
    mask = (mask_x.unsqueeze(1) & mask_y.unsqueeze(2)).unsqueeze(1)  # [B, 1, H, W]

    # Apply CutMix directly with the mask
    cutmixed = real_batch * (~mask) + target_b * mask

    # Adjust ratios to match pixel ratio for each image
    bbox_area = (bbx2 - bbx1) * (bby2 - bby1)
    ratios = 1 - (bbox_area / (W * H))

    # Generate CutMix for the decoded outputs
    _, D_g_decoder = D_unet(target_b)
    _, D_decoder = D_unet(target_a)

    cutmixed_decoded = D_decoder * (~mask) + D_g_decoder * mask

    return ratios, cutmixed, cutmixed_decoded, target_a, target_b, bbx1, bbx2, bby1, bby2


def rand_bbox(size, ratios,device):
    batch_size, _, H, W = size
    # Compute bounding box for each image based on its ratio
    cut_ratios = torch.sqrt(1 - ratios).to(device)  # sqrt for CutMix formula

    bbx1 = torch.randint(0, W, (batch_size,), device=device)
    bby1 = torch.randint(0, H, (batch_size,), device=device)

    bbx2 = torch.clamp(bbx1 + (cut_ratios * W).long(), max=W)
    bby2 = torch.clamp(bby1 + (cut_ratios * H).long(), max=H)

    return bbx1, bby1, bbx2, bby2


def loss_encoder(output, labels):
    loss = F.binary_cross_entropy(output, labels, reduction='mean')
    return loss

def loss_decoder(output, labels):
    loss = F.binary_cross_entropy(output, labels, reduction='mean')
    return loss

def loss_regularization(output, target):
    loss = F.pairwise_distance(output, target, p=2, keepdim=False).mean()
    return loss


