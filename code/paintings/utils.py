import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils import spectral_norm
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import itertools



class PaintingsDataset(Dataset):
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


class Generator(nn.Module):
    def __init__(self, latent_dim, channels_out):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            # Upscale to 4x4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Upscale to 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Upscale to 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Upscale to 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Upscale to 64x64
            nn.ConvTranspose2d(64, channels_out, 4, 2, 1, bias=False),
            nn.Tanh()  # Output: channels_out x 128 x 128
        )

        # Final layer to upscale to 128x128
        self.final_layer = nn.ConvTranspose2d(64, channels_out, 4, 2, 1, bias=False)

    def forward(self, x):
        x = self.model(x)
        # x = self.final_layer(x)
        return x

    

class Discriminator(nn.Module):
    def __init__(self, channels_in):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: (channels_in, 256, 256)
            nn.Conv2d(channels_in, 64, 4, 2, 1, bias=False),  # Output: (64, 128, 128)
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # Output: (128, 64, 64)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # Output: (256, 32, 32)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # Output: (512, 16, 16)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=8, stride=1, padding=0, bias=False) # Output: (1, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(1, 1),  # Convert the final output to a single scalar
            nn.Sigmoid()  # Output probability
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)  # Flatten the output to (batch_size, 1)
        return self.fc(x)


class DoubleConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.doubleConv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.doubleConv(x)
    

class DownSample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = DoubleConv(input_channels, output_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv(x)
        return x1, self.max_pool(x1)

# class UpSample(nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super().__init__()
#         self.Upsample = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2)
#         self.conv = DoubleConv(input_channels, output_channels)

#     def forward(self, x1, x2):
#         x1 = self.Upsample(x1)
#         x = torch.cat([x1,x2], 1)
#         return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x1, x2], 1)

        return self.conv(x)



# class Unet_Discriminator(nn.Module):
#     def __init__(self, input_channels, n_classes):
#         super().__init__()
#         # Input size: (3, 64, 64)
#         self.down_1 = DownSample(input_channels, 64)
#         # Input size: (64, 32, 32)
#         self.down_2 = DownSample(64, 128)
#         # Input size: (128, 16, 16)
#         self.down_3 = DownSample(128, 256)
#         # Input size: (256, 8, 8)
#         self.down_4 = DownSample(256, 512)
#         # Input size: (512, 4, 4)
#         self.down_5 = DownSample(512, 1024)
#         # Input size: (1024, 2, 2)
#         self.down_6 = DownSample(1024, 2048) # Output size: (2048, 1, 1)

#         self.fc1 = nn.Linear(2048 * 1, 1, bias=False)
#         self.activation_1 = nn.Sigmoid()

#         self.bottle_neck = DoubleConv(2048, 2048)

#         # Input size: (2048, 1, 1)
#         self.up_1 = UpSample(2048, 1024)
#         # Input size: (1024, 2, 2)
#         self.up_2 = UpSample(1024, 512)
#         # Input size: (512, 4, 4)
#         self.up_3 = UpSample(512, 256)
#         # Input size: (256, 8, 8)
#         self.up_4 = UpSample(256, 128)
#         # Input size: (128, 16, 16)
#         self.up_5 = UpSample(128, 64)
#         # Input size: (64, 32, 32)
#         self.up_6 = UpSample(64, 32) # Output size: (32, 64, 64)

#         self.output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1) 

#     def forward(self, x):
#         down1, p1 = self.down_1(x)
#         down2, p2 = self.down_2(p1)
#         down3, p3 = self.down_3(p2)
#         down4, p4 = self.down_4(p3)
#         down5, p5 = self.down_5(p4)
#         down6, p6 = self.down_6(p5)

#         p6_pooled = torch.sum(p6, dim=[2,3])
#         out_1 = self.fc1(p6_pooled)
#         out_1 = self.activation_1(out_1)
#         b = self.bottle_neck(p6)


#         up1 = self.up_1(b, down6)
#         up2 = self.up_2(up1, down5)
#         up3 = self.up_3(up2, down4)
#         up4 = self.up_4(up3, down3)
#         up5 = self.up_5(up4, down2)
#         up6 = self.up_6(up5, down1)

#         out_2 = self.output(up6)
#         out_2 = self.activation_1(out_2)

#         return out_1, out_2
    
# class Unet_Discriminator(nn.Module):
    # def __init__(self, input_channels, n_classes):
    #     super().__init__()
    #     # Input size: (3, 64, 64)
    #     self.down_1 = DownSample(input_channels, 64)
    #     # Input size: (64, 32, 32)
    #     self.down_2 = DownSample(64, 128)
    #     # Input size: (128, 16, 16)
    #     self.down_3 = DownSample(128, 256)
    #     # Input size: (256, 8, 8)
    #     self.down_4 = DownSample(256, 512)
    #     # Input size: (512, 4, 4)
    #     self.down_5 = DownSample(512, 1024)
    #     # Input size: (1024, 2, 2)
    #     self.down_6 = DownSample(1024, 2048)  # Output size: (2048, 1, 1)

    #     self.fc1 = nn.Linear(2048 * 1, 1, bias=False)
    #     self.activation_1 = nn.Sigmoid()

    #     self.bottle_neck = DoubleConv(2048, 2048)

    #     # Input size: (2048, 1, 1)
    #     self.up_1 = UpSample(4096, 1024)
    #     # Input size: (1024, 2, 2)
    #     self.up_2 = UpSample(2048, 1024)
    #     # Input size: (512, 4, 4)
    #     self.up_3 = UpSample(1024, 512)
    #     # Input size: (256, 8, 8)
    #     self.up_4 = UpSample(512, 256)
    #     # Input size: (128, 16, 16)
    #     self.up_5 = UpSample(256, 128)
    #     # Input size: (64, 32, 32)
    #     self.up_6 = UpSample(128, 64)  # Output size: (32, 64, 64)

    #     self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1) 

    # def forward(self, x):
    #     down1, p1 = self.down_1(x)
    #     down2, p2 = self.down_2(p1)
    #     down3, p3 = self.down_3(p2)
    #     down4, p4 = self.down_4(p3)
    #     down5, p5 = self.down_5(p4)
    #     down6, p6 = self.down_6(p5)

    #     p6_pooled = torch.sum(p6, dim=[2,3])
    #     out_1 = self.fc1(p6_pooled)
    #     out_1 = self.activation_1(out_1)
    #     b = self.bottle_neck(p6)

    #     print(b.shape)
    #     print(down5.shape)
    #     # Correcting the upsampling channels:
    #     up1 = self.up_1(b, down6)  # Concatenate b and down6
    #     up2 = self.up_2(up1, down5)  # Concatenate up1 and down5
    #     up3 = self.up_3(up2, down4)  # Concatenate up2 and down4
    #     up4 = self.up_4(up3, down3)  # Concatenate up3 and down3
    #     up5 = self.up_5(up4, down2)  # Concatenate up4 and down2
    #     up6 = self.up_6(up5, down1)  # Concatenate up5 and down1

    #     out_2 = self.output(up6)
    #     out_2 = self.activation_1(out_2)

    #     return out_1, out_2

class Unet_Discriminator(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(Unet_Discriminator, self).__init__()

        # Downsample layers
        self.down_1 = DownSample(input_channels, 64)    # (64, 32, 32)
        self.down_2 = DownSample(64, 128)               # (128, 16, 16)
        self.down_3 = DownSample(128, 256)              # (256, 8, 8)
        self.down_4 = DownSample(256, 512)              # (512, 4, 4)
        # self.down_5 = DownSample(512, 1024)             # (1024, 2, 2)
        # self.down_6 = DownSample(1024, 2048)            # (2048, 1, 1)

        # self.fc1 = nn.Linear(2048 * 1 * 1, 1, bias=False)  # Fully connected layer for classification
        self.fc1 = nn.Linear(512 * 1 * 1, 1, bias=False)
        self.activation_1 = nn.Sigmoid()

        # self.bottle_neck = DoubleConv(2048, 2048)
        self.bottle_neck = DoubleConv(512, 512)

        # Upsample layers
        # self.up_1 = UpSample(512 + 2048, 1024)    # (1024, 2, 2)
        # self.up_2 = UpSample(1024 + 1024, 512)     # (512, 4, 4)
        # self.up_3 = UpSample(512 + 512, 256)       # (256, 8, 8)
        # self.up_4 = UpSample(256 + 256, 128)       # (128, 16, 16)
        # self.up_5 = UpSample(128 + 128, 64)        # (64, 32, 32)
        # self.up_6 = UpSample(64 + 64, 32)          # (32, 64, 64)
        self.up_1 = UpSample(512 + 512, 256)       # (256, 8, 8)
        self.up_2 = UpSample(256 + 256, 128)       # (128, 16, 16)
        self.up_3 = UpSample(128 + 128, 64)        # (64, 32, 32)
        self.up_4 = UpSample(64 + 64, 32)          # (32, 64, 64)

        self.output = nn.Conv2d(32, out_channels=1, kernel_size=1)

    def forward(self, x):
        down1, p1 = self.down_1(x)
        down2, p2 = self.down_2(p1)
        down3, p3 = self.down_3(p2)
        down4, p4 = self.down_4(p3)
        # down5, p5 = self.down_5(p4)
        # down6, p6 = self.down_6(p5)

        # p6_pooled = torch.sum(p6, dim=[2, 3])
        # out_1 = self.fc1(p6_pooled)
        # out_1 = self.activation_1(out_1)
        # b = self.bottle_neck(p6)
        p4_pooled = torch.sum(p4, dim=[2, 3])
        out_1 = self.fc1(p4_pooled)
        out_1 = self.activation_1(out_1)
        b = self.bottle_neck(p4)

        # up1 = self.up_1(b, down6)
        # up2 = self.up_2(up1, down5)
        # up3 = self.up_3(up2, down4)
        # up4 = self.up_4(up3, down3)
        up1 = self.up_1(b, down4)
        up2 = self.up_2(up1, down3)
        up3 = self.up_3(up2, down2)
        up4 = self.up_4(up3, down1)
        # up5 = self.up_5(up4, down2)
        # up6 = self.up_6(up5, down1)

        out_2 = self.output(up4)
        out_2 = self.activation_1(out_2)
        return out_1, out_2




# class Unet_Generator(nn.Module):
#     def __init__(self, latent_dim, channels_out=3):
#         super(Unet_Generator, self).__init__()

#         self.initial_layer = nn.Sequential(
#             # Input: latent_dim x 1 x 1
#             nn.Conv2d(latent_dim, 512, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU()
#         )

#         self.upsample_blocks = nn.ModuleList([
#             # Upscale to 2x2
#             self._upsample_block(512, 256),
#             # Upscale to 4x4
#             self._upsample_block(256, 128),
#             # Upscale to 8x8
#             self._upsample_block(128, 64),
#             # Upscale to 16x16
#             self._upsample_block(64, 32),
#             # Upscale to 32x32
#             self._upsample_block(32, 16),
#             # Upscale to 64x64
#             self._upsample_block(16, channels_out, final_block=True)
#         ])

#     def _upsample_block(self, in_channels, out_channels, final_block=False):
#         """Helper function to create an upsample block."""
#         layers = [
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#         ]
#         if not final_block:
#             layers += [
#                 nn.BatchNorm2d(out_channels),
#                 nn.LeakyReLU(0.2),
#                 nn.Dropout(0.2)
#             ]
#         else:
#             layers.append(nn.Tanh())
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         # Reshape latent vector for initial layer
#         x = x.view(x.size(0), x.size(1), 1, 1)
#         x = self.initial_layer(x)

#         # Pass through upsample blocks
#         for block in self.upsample_blocks:
#             x = block(x)
        
#         return x


class Unet_Generator(nn.Module):
    def __init__(self, latent_dim, channels_out=3, base_channels=64, num_upsamples=6):
        super(Unet_Generator, self).__init__()

        self.base_channels = base_channels
        self.num_upsamples = num_upsamples

        # Downsampling blocks
        self.downsample_blocks = nn.ModuleList()
        in_channels = latent_dim
        for i in range(num_upsamples):
            out_channels = base_channels * (2 ** i)
            self.downsample_blocks.append(self._downsample_block(in_channels, out_channels))
            in_channels = out_channels

        in_channels *= 2

        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        for i in range(num_upsamples - 1, -1, -1):
            out_channels = base_channels * (2 ** (i))
            self.upsample_blocks.append(self._upsample_block(in_channels, out_channels//2))
            in_channels = out_channels

        # Final layer to generate output image
        self.final_block = nn.Sequential(
            nn.Conv2d(base_channels//2, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def _downsample_block(self, in_channels, out_channels):
        """Downsampling block with stride-2 convolution."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _upsample_block(self, in_channels, out_channels):
        """Upsampling block with ConvTranspose2d."""
        return nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Downsampling path
        skips = []
        for block in self.downsample_blocks:
            x = block(x)
            skips.append(x)

        # Reverse the skip connections for the upsampling path
        skips = skips[::-1]
        
        # Upsampling path
        for i, block in enumerate(self.upsample_blocks):
           if i < len(skips):  # Add skip connection 
                x = torch.cat([x, skips[i]], dim=1)
                x = block(x)

        # Final output
        x = self.final_block(x)
        return x
        
class Unet_Discriminator_V2(nn.Module):
    def __init__(self, input_channels, n_classes):
        super().__init__()

        self.down_1 = DownSample(input_channels, 64)
        self.down_2 = DownSample(64, 128)
        self.down_3 = DownSample(128, 256)

        self.fc1 = nn.Linear(256 * 1, 1, bias=False)
        self.activation_1 = nn.Sigmoid()

        self.bottle_neck = DoubleConv(256, 512)

        self.up_1 = UpSample(512, 256)
        self.up_2 = UpSample(256, 128)
        self.up_3 = UpSample(128, 64)

        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1) 

    def forward(self, x):
        down1, p1 = self.down_1(x)
        down2, p2 = self.down_2(p1)
        down3, p3 = self.down_3(p2)

        p3_pooled = torch.sum(p3, dim=[2,3])
        out_1 = self.fc1(p3_pooled)
        out_1 = self.activation_1(out_1)
        b = self.bottle_neck(p3)


        up1 = self.up_1(b, down3)
        up2 = self.up_2(up1, down2)
        up3 = self.up_3(up2, down1)

        out_2 = self.output(up3)
        out_2 = self.activation_1(out_2)

        return out_1, out_2
    
class Unet_Generator_V2(nn.Module):
    def __init__(self, latent_dim, channels_out):
        super(Unet_Generator_V2, self).__init__()
        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 64, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Upscale to 4x4
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Upscale to 16x16
            nn.ConvTranspose2d(32, channels_out, 4, 2, 1, bias=False),
            nn.Tanh()  # Output: channels_out x 16 x 16
        )

        # Final layer to upscale to 256x256
        # self.final_layer = nn.ConvTranspose2d(channels_out, channels_out, 4, 2, 1, bias=False)

    def forward(self, x):
        x = self.model(x)
        # x = self.final_layer(x)
        return x



    

# class Unet_Generator(nn.Module):
#     def __init__(self, latent_dim, channels_out):
#         super(Unet_Generator, self).__init__()
#         self.model = nn.Sequential(
#             # Input: latent_dim x 1 x 1
#             nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),

#             # Upscale to 4x4
#             nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),

#             # Upscale to 8x8
#             nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             # Upscale to 16x16
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             # Upscale to 32x32
#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             # Upscale to 64x64
#             nn.ConvTranspose2d(64, channels_out, 4, 2, 1, bias=False),
#             nn.Tanh()  # Output: channels_out x 128 x 128
#         )

#         # Final layer to upscale to 256x256
#         # self.final_layer = nn.ConvTranspose2d(channels_out, channels_out, 4, 2, 1, bias=False)

    # def forward(self, x):
    #     x = self.model(x)
    #     # x = self.final_layer(x)
    #     return x
    
def unet_d_criterion_without_cutmix(output, label, batch_size):
    out_1, out_2 = output
    label_2 = label.view(batch_size, 1, 1, 1)
    label_2 = label_2.expand(-1, 1, 64, 64)

    out_1 = torch.clamp(out_1, 1e-10, 1 - 1e-10)
    out_2 = torch.clamp(out_2, 1e-10, 1 - 1e-10)

    loss_1 = F.binary_cross_entropy(out_1, label, reduction='mean')
    loss_2 = F.binary_cross_entropy(out_2, label_2, reduction='mean')

    return loss_1, loss_2



def unet_d_criterion_with_cutmix(output, M, batch_size, epsilon=1e-10):
    out_1, out_2 = output

    # Ensure values are within a range to prevent log(0)
    out_1 = torch.clamp(out_1, min=epsilon, max=1-epsilon)
    out_2 = torch.clamp(out_2, min=epsilon, max=1-epsilon)
    
    # Compute log values
    loss_1 = -torch.sum(torch.log(out_1)) 

    p1 = out_2 * M
    p2 = out_2 * (1 - M)
    
    p1 = torch.clamp(p1, min=epsilon, max=1-epsilon)
    p2 = torch.clamp(p2, min=epsilon, max=1-epsilon)
    
    # Loss computation for p1 and p2
    loss_2 = -torch.sum(M * torch.log(p1) + (1 - M) * torch.log(p2))

    return (loss_1 + loss_2) / batch_size


# def rand_bbox(size, lam):
#     W = size[-2]
#     H = size[-1]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = int(W * cut_rat)
#     cut_h = int(H * cut_rat)

#     # uniform
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)

#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)

#     return bbx1, bby1, bbx2, bby2


W, H = 128, 128



# def generate_CutMix_samples(real_batch, fake_batch, D_unet):
#     # generate mixed sample
#     ratio = np.random.rand()
#     size = real_batch.size()
#     W, H = size[2], size[3]

#     batch_size = size[0]
#     rand_indices = torch.randperm(batch_size)

#     target_a = real_batch.clone()  # Clone real images
#     target_b = fake_batch[rand_indices].clone()  # Clone shuffled fake images

#     # Generate random bounding box
#     bbx1, bby1, bbx2, bby2 = rand_bbox(size, ratio)

#     # Generate the mask for CutMix
#     mask = torch.ones_like(real_batch)
#     mask[:, :, bbx1:bbx2, bby1:bby2] = 0  # Masking the mixed region

#     # Use torch.where to apply the CutMix without in-place modification
#     cutmixed = torch.where(mask == 1, target_a, target_b)

#     # adjust lambda to exactly match pixel ratio
#     ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

#     # Generate cutmix that will be used in the loss
#     D_decoder, D_g_decoder = D_unet(target_a)[1], D_unet(target_b)[1]

#     # Use torch.where to apply the CutMix in decoded results
#     cutmixed_decoded = torch.where(mask == 1, D_decoder, D_g_decoder)

#     return ratio, cutmixed, cutmixed_decoded, target_a, target_b, bbx1, bbx2, bby1, bby2



def generate_CutMix_samples(real_batch, fake_batch, D_unet, device=torch.device('cpu')):
    batch_size, _, H, W = real_batch.size()

    # Generate random ratios for the batch
    ratios = torch.rand(batch_size, device=device)

    # Randomly permute the fake batch for CutMix
    rand_indices = torch.randperm(batch_size, device=device)

    target_a = real_batch.clone()
    target_b = fake_batch[rand_indices].clone()

    # Generate bounding boxes for each image in the batch
    bbx1, bby1, bbx2, bby2 = rand_bbox(real_batch.size(), ratios, device)

    # Apply CutMix using batch indexing
    cutmixed = real_batch.clone()
    for i in range(batch_size):
        cutmixed[i, :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = target_b[i, :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]]

    # Adjust ratios to match pixel ratio for each image
    ratios = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    # Generate CutMix for the decoded outputs
    D_decoder, D_g_decoder = D_unet(target_a)[1], D_unet(target_b)[1]
    cutmixed_decoded = D_decoder.clone()
    for i in range(batch_size):
        cutmixed_decoded[i, :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = D_g_decoder[i, :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]]

    return ratios, cutmixed, cutmixed_decoded, target_a, target_b, bbx1, bbx2, bby1, bby2

# def generate_CutMix_samples(real_batch, fake_batch, D_unet, device):
#     # Ensure inputs are valid

#     batch_size, _, H, W = real_batch.size()

#     # Generate random ratios for the batch
#     ratios = torch.rand(batch_size, device=device)

#     # Randomly permute the fake batch for CutMix
#     rand_indices = torch.randperm(batch_size, device=device)

#     # Prepare targets
#     target_a = real_batch
#     target_b = fake_batch[rand_indices]

#     # Generate bounding boxes for each image in the batch
#     bbx1, bby1, bbx2, bby2 = rand_bbox(real_batch.size(), ratios, device)

#     # Apply CutMix using vectorized operations
#     cutmixed = real_batch.clone()
#     cutmixed[:, :, bbx1:bbx2, bby1:bby2] = target_b[:, :, bbx1:bbx2, bby1:bby2]

#     # Adjust ratios to match pixel ratio for each image
#     ratios = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

#     # Generate CutMix for the decoded outputs
#     combined_inputs = torch.cat([target_a, target_b], dim=0)
#     combined_outputs = D_unet(combined_inputs)
#     D_decoder = combined_outputs[1][:batch_size]
#     D_g_decoder = combined_outputs[1][batch_size:]

#     cutmixed_decoded = D_decoder.clone()
#     cutmixed_decoded[:, :, bbx1:bbx2, bby1:bby2] = D_g_decoder[:, :, bbx1:bbx2, bby1:bby2]

#     return ratios, cutmixed, cutmixed_decoded, target_a, target_b, bbx1, bbx2, bby1, bby2


def rand_bbox(size, ratios,device):
    batch_size, _, H, W = size
    # Compute bounding box for each image based on its ratio
    cut_ratios = torch.sqrt(1 - ratios).to(device)  # sqrt for CutMix formula

    bbx1 = torch.randint(0, W, (batch_size,), device=device)
    bby1 = torch.randint(0, H, (batch_size,), device=device)

    bbx2 = torch.clamp(bbx1 + (cut_ratios * W).long(), max=W)
    bby2 = torch.clamp(bby1 + (cut_ratios * H).long(), max=H)

    return bbx1, bby1, bbx2, bby2




def mix(M, G, x):
    mixed = M * x + (1 - M) * G
    return mixed 


def loss_encoder(output, labels):
    loss = F.binary_cross_entropy(output, labels, reduction='sum')
    return loss

def loss_decoder(output, labels):
    loss = F.binary_cross_entropy(output, labels, reduction='sum')
    return loss

def loss_regularization(output, target):
    loss = F.pairwise_distance(output, target, p=2, keepdim=False).sum()
    return loss


import torch
import torch.nn as nn
import torch.optim as optim

# Generator
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, g_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=2, stride=2, bias=False)
        self.phi = nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        sum_features = theta_x + phi_g
        psi = self.sigmoid(self.psi(sum_features))
        upsampled_psi = self.upsample(psi)
        return x * upsampled_psi

class AttentionUNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, features=64):
        super(AttentionUNetGenerator, self).__init__()
        self.encoder1 = self._block(in_channels, features)
        self.encoder2 = self._block(features, features * 2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.encoder4 = self._block(features * 4, features * 8)
        
        self.bottleneck = self._block(features * 8, features * 16)
        
        self.attention4 = AttentionBlock(features * 8, features * 16, features * 4)
        self.attention3 = AttentionBlock(features * 4, features * 8, features * 2)
        self.attention2 = AttentionBlock(features * 2, features * 4, features)
        
        self.up4 = self._block(features * 16, features * 8)
        self.up3 = self._block(features * 8, features * 4)
        self.up2 = self._block(features * 4, features * 2)
        self.up1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, kernel_size=1)
        )
        self.upconv4 = self._upsample(features * 16, features * 8)
        self.upconv3 = self._upsample(features * 8, features * 4)
        self.upconv2 = self._upsample(features * 4, features * 2)
        self.upconv1 = self._upsample(features * 2, features)


    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(nn.MaxPool2d(2)(e1))
        e3 = self.encoder3(nn.MaxPool2d(2)(e2))
        e4 = self.encoder4(nn.MaxPool2d(2)(e3))
        
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(e4))
        
        a4 = self.attention4(e4, bottleneck)
        up4 = self.upconv4(bottleneck)
        up4 = self.up4(torch.cat([up4, a4], dim=1))
        
        a3 = self.attention3(e3, up4)
        up3 = self.upconv3(up4)
        up3 = self.up3(torch.cat([up3, a3], dim=1))
        
        a2 = self.attention2(e2, up3)
        up2 = self.upconv2(up3)
        up2 = self.up2(torch.cat([up2, a2], dim=1))
        
        up1 = self.upconv1(up2)
        out = self.up1(torch.cat([up1, e1], dim=1))
        
        return torch.tanh(out)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

class AttentionBlock2(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock2, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scalar weight

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute query, key, and value
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # [B, H*W, C/8]
        key = self.key(x).view(batch_size, -1, height * width)  # [B, C/8, H*W]
        value = self.value(x).view(batch_size, -1, height * width)  # [B, C, H*W]

        # Compute attention map
        attention = torch.bmm(query, key)  # [B, H*W, H*W]
        attention = torch.softmax(attention, dim=-1)

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, H*W]
        out = out.view(batch_size, channels, height, width)  # Reshape to [B, C, H, W]

        # Add skip connection
        out = self.gamma * out + x
        return out

class AttentionUNetDiscriminator(nn.Module):
    def __init__(self, in_channels, features=64, use_sigmoid=True):
        super(AttentionUNetDiscriminator, self).__init__()
        self.encoder1 = self._block(in_channels, features)
        self.attention1 = AttentionBlock2(features)
        self.encoder2 = self._block(features, features * 2)
        self.attention2 = AttentionBlock2(features * 2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.attention3 = AttentionBlock2(features * 4)
        self.encoder4 = self._block(features * 4, features * 8)
        self.attention4 = AttentionBlock2(features * 8)
        self.final = spectral_norm(nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=0))
        self.use_sigmoid = use_sigmoid
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.attention3(self.encoder3(e2))
        e4 = self.encoder4(e3)
        out = self.final(e4)
        if self.use_sigmoid:
            return torch.sigmoid(out)
        else:
            return out

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        
if __name__=="__main__":
    # input_image = torch.randn(64, 3, 64, 64)  # Batch size of 16, RGB channels, 64x64 resolution
    # D = AttentionUNetDiscriminator(in_channels=3)
    # output = D(input_image)
    
    # print(output.shape)  # Expected output: torch.Size([16, 1, 1, 1])
    
    # latent_dim = 40
    # noise = torch.randn(16, latent_dim, 64, 64)
    # G = AttentionUNetGenerator(in_channels=40, out_channels=3)
    # output = G(noise)
    
    # print(output.shape)  # Expected output: torch.Size([16, 3, 64, 64])
    latent_dim = 40
    output_channels = 3
    noise = torch.randn(16, latent_dim, 64, 64)
    G = Unet_Generator(latent_dim, num_upsamples=4)
    
    output = G(noise)
    print(output.shape)  # Expected output: torch.Size([16, 3, 64, 64])
    
    for name, param in G.named_parameters():
        if "weight" in name:
            print(name, param.mean().item(), param.std().item())
            
    print(output[0][0])