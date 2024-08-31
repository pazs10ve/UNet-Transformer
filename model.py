import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size = patch_size, stride = patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    


"""
in_channels = 4
embed_dim = 768
patch_size = 16
patch_embedding = PatchEmbedding(in_channels, embed_dim, patch_size)
b = 1
img_size = (96, 96, 96)
c = 4
x = torch.randn(b, c, *img_size)
patch_embedding(x)

torch.Size([1, 4, 96, 96, 96])
torch.Size([1, 768, 6, 6, 6])
torch.Size([1, 216, 768])
"""


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio = 4):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads, dim_feedforward = embed_dim * mlp_ratio)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

"""
embed_dim = 768
num_heads = 12
num_layers = 12
b = 1
seq_len = 216

transformer = TransformerEncoder(embed_dim, num_heads, num_layers)
x = torch.randn(b, seq_len, embed_dim)
transformer(x)
torch.Size([1, 216, 768])
torch.Size([1, 216, 768])
torch.Size([1, 216, 768])
torch.Size([1, 216, 768])
torch.Size([1, 216, 768])
torch.Size([1, 216, 768])
torch.Size([1, 216, 768])
torch.Size([1, 216, 768])
torch.Size([1, 216, 768])
torch.Size([1, 216, 768])
torch.Size([1, 216, 768])
torch.Size([1, 216, 768])
torch.Size([1, 216, 768])

"""


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

"""
b = 1
in_channels = 64
out_channels = 64
img_size = (96, 96, 96)

conv_block = ConvBlock(in_channels, out_channels)
x = torch.randn(b, in_channels, *img_size)
conv_block(x)
torch.Size([1, 64, 96, 96, 96])
torch.Size([1, 64, 96, 96, 96])
"""

class MiniDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiniDeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size = 2, stride = 2)

    def forward(self, x):
        x = self.deconv(x)
        return x
    

"""
in_channels = 128
out_channels = 64
b = 1
img_size = (48, 48, 48)
mini_deconv_block = MiniDeconvBlock(in_channels, out_channels)
x = torch.randn(b, in_channels, *img_size)
mini_deconv_block(x)
torch.Size([1, 128, 48, 48, 48])
torch.Size([1, 64, 96, 96, 96])
"""

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size = 2, stride = 2),
            nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.deconv(x)
        return x


"""
in_channels = 128
out_channels = 64
b= 1
img_size = (32, 32, 32)
deconv_block = DeconvBlock(in_channels, out_channels)
x = torch.randn(b, in_channels, *img_size)
deconv_block(x)
torch.Size([1, 128, 32, 32, 32])
torch.Size([1, 64, 64, 64, 64])

"""

class UNETR(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, patch_size, embed_dim, num_heads, num_layers):
        super(UNETR, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers)


        """
        HxWxDx4 -> HxWxDx64
        """
        self.conv1 = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 64)
        )


        """
        H/2xW/2xD/2x128 -> H/2xW/2xD/2x128
        """
        self.conv2 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )


        """
        H/4xW/4xD/4x256 -> H/4xW/4xD/4x256
        """
        self.conv3 = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )


        """
        H/8xW/8xD/8x512 -> H/8xW/8xD/8x512
        """
        self.conv4 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )


        """
        HxWxDx64 -> HxWxDx3
        """
        self.final_conv = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            nn.Conv3d(64, out_channels, kernel_size=1)
        )

        """
        H/2xW/2xD/2x128 -> HxWxDx64
        """
        self.deconv1 = MiniDeconvBlock(128, 64)


        """
        H/4xW/4xD/4x256 -> H/2xW/2xD/2x128
        """
        self.deconv2 = MiniDeconvBlock(256, 128)

        """
        H/8xW/8xD/8x512 -> H/4xW/4xD/4x256
        """
        self.deconv3 = MiniDeconvBlock(512, 256)


        """
        H/16xW/16xD/16x768 -> H/8xW/8xD/8x512
        """
        self.deconv4 = MiniDeconvBlock(768, 512)


        """
        H/16xW/16xD/16x768 -> H/2xW/2xD/2x128
        """
        self.decoder1 = nn.Sequential(
            DeconvBlock(768, 512),
            DeconvBlock(512, 256),
            DeconvBlock(256, 128)
        )


        """
        H/16xW/16xD/16x768 -> H/4xW/4xD/4x256
        """
        self.decoder2 = nn.Sequential(
            DeconvBlock(768, 512),
            DeconvBlock(512, 256)
        )


        """
        H/16xW/16xD/16x768 -> H/8xW/8xD/8x512
        """
        self.decoder3 = nn.Sequential(
            DeconvBlock(768, 512)
        )

    def forward(self, x):
        B, C, H, W, D = x.shape

        """
        Image dimensions must be divisible by the patch size
        """
        assert H % self.patch_size == 0 and W % self.patch_size == 0 and D % self.patch_size == 0

        x0 = self.conv1(x)

        x = self.patch_embedding(x)
        features = self.transformer(x)

        z12, z9, z6, z3 = features[-1], features[-4], features[-7], features[-10]

        z12 = z12.transpose(1, 2).view(B, -1, H//self.patch_size, W//self.patch_size, D//self.patch_size)
        z9 = z9.transpose(1, 2).view(B, -1,  H//self.patch_size, W//self.patch_size, D//self.patch_size)
        z6 = z6.transpose(1, 2).view(B, -1,  H//self.patch_size, W//self.patch_size, D//self.patch_size)
        z3 = z3.transpose(1, 2).view(B, -1, H//self.patch_size, W//self.patch_size, D//self.patch_size)

        x = self.deconv4(z12)
        z9 = self.decoder3(z9)
        x = x + z9

        z6 = self.decoder2(z6)
        x = self.conv4(x)
        x = self.deconv3(x)
        x = x + z6

        z3 = self.decoder1(z3)
        x = self.conv3(x)
        x = self.deconv2(x)
        x = x + z3

        x = self.conv2(x)
        x = self.deconv1(x)
        x = x + x0

        x = self.final_conv(x)
        return x

        



"""
batch_size = 1
in_channels = 1
out_channels = 1
img_size = (96, 96, 96)
patch_size = 16
embed_dim = 768
num_heads = 12
num_layers = 12

model = UNETR(in_channels, out_channels, img_size, patch_size, embed_dim, num_heads, num_layers)

x = torch.randn(batch_size, in_channels, *img_size)
out = model(x)
print(out.shape)
torch.Size([1, 1, 96, 96, 96])
"""


def get_model(in_channels : int, out_channels : int, img_size : Any = (96, 96, 96), patch_size : int = 16, embed_dim : int = 768, num_heads : int = 12, num_layers : int = 12):
    model = UNETR(in_channels, out_channels, img_size, patch_size, embed_dim, num_heads, num_layers)
    return model

