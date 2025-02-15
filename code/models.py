import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


##### CutMix GAN Model #####

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


class UpSample(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x1, x2], 1)

        return self.conv(x)


class Unet_Discriminator(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(Unet_Discriminator, self).__init__()

        # Downsample layers
        self.down_1 = DownSample(input_channels, 64)    # (64, 32, 32)
        self.down_2 = DownSample(64, 128)               # (128, 16, 16)
        self.down_3 = DownSample(128, 256)              # (256, 8, 8)
        self.down_4 = DownSample(256, 512)              # (512, 4, 4)

        self.fc1 = nn.Linear(512 * 1 * 1, 1, bias=False)
        self.activation_1 = nn.Sigmoid()

        self.bottle_neck = DoubleConv(512, 512)

        # Upsample layers
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
       
        p4_pooled = torch.sum(p4, dim=[2, 3])
        out_1 = self.fc1(p4_pooled)
        out_1 = self.activation_1(out_1)
        b = self.bottle_neck(p4)

        up1 = self.up_1(b, down4)
        up2 = self.up_2(up1, down3)
        up3 = self.up_3(up2, down2)
        up4 = self.up_4(up3, down1)

        out_2 = self.output(up4)
        out_2 = self.activation_1(out_2)
        return out_1, out_2


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
    
    
#### Attention WGAN-GP Model ####

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
        