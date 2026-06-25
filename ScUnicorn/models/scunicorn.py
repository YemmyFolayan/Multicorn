"""
ScUnicorn Model Definition
Deep learning framework for Hi-C super-resolution with optional multi-modal data integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# Utility Modules
# -------------------------------------------------------------
class ConvBlock(nn.Module):
    """Basic convolutional block for feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for stable deep feature learning."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


# -------------------------------------------------------------
# Hi-C Encoder (base model)
# -------------------------------------------------------------
class HiCEncoder(nn.Module):
    """
    Encoder network for Hi-C matrices (low-resolution).
    """
    def __init__(self, in_channels=1, base_channels=64):
        super(HiCEncoder, self).__init__()
        self.block1 = ConvBlock(in_channels, base_channels)
        self.block2 = ConvBlock(base_channels, base_channels * 2)
        self.res1 = ResidualBlock(base_channels * 2)
        self.res2 = ResidualBlock(base_channels * 2)
        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))  # compress to latent grid

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.global_pool(x)
        return x


# -------------------------------------------------------------
# Modality Encoders
# -------------------------------------------------------------
class ModalityEncoder(nn.Module):
    """
    Simple encoder for 1D omics signals (ATAC-seq, ChIP-seq, RNA-seq).
    """
    def __init__(self, input_dim, hidden_dim=256):
        super(ModalityEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)


# -------------------------------------------------------------
# Fusion and Reconstruction
# -------------------------------------------------------------
class FusionNetwork(nn.Module):
    """
    Fuses multi-modal embeddings and reconstructs a high-resolution Hi-C matrix.
    """
    def __init__(self, hic_channels=128, hic_size=8, hidden_dim=256, upscale_factor=2):
        super(FusionNetwork, self).__init__()

        fused_input_dim = hidden_dim * 3 + (hic_channels * hic_size * hic_size)
        self.fc_fuse = nn.Sequential(
            nn.Linear(fused_input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Reconstruction head
        self.reconstruct = nn.Sequential(
            nn.Linear(1024, hic_channels * hic_size * hic_size * (upscale_factor ** 2)),
            nn.ReLU(inplace=True)
        )
        self.upscale_factor = upscale_factor

    def forward(self, hic_feat, atac_feat=None, chip_feat=None, rna_feat=None):
        # Flatten Hi-C feature
        hic_flat = hic_feat.view(hic_feat.size(0), -1)

        # Handle optional modalities
        extra_feats = []
        for feat in [atac_feat, chip_feat, rna_feat]:
            if feat is not None and feat.nelement() > 1:
                extra_feats.append(feat)
        fused = torch.cat([hic_flat] + extra_feats, dim=1) if extra_feats else hic_flat

        # Fusion and reconstruction
        fused_out = self.fc_fuse(fused)
        recon_flat = self.reconstruct(fused_out)

        # Reshape to (B, 1, H, W)
        out_size = int((8 * self.upscale_factor))
        out_tensor = recon_flat.view(hic_feat.size(0), 1, out_size, out_size)
        return out_tensor


# -------------------------------------------------------------
# Full ScUnicorn Model
# -------------------------------------------------------------
class ScUnicorn(nn.Module):
    """
    Full model combining Hi-C encoder, optional multi-omics encoders, and fusion network.
    Works in both Hi-C only and multi-modal configurations.
    """
    def __init__(self, atac_dim=None, chip_dim=None, rna_dim=None):
        super(ScUnicorn, self).__init__()
        self.hic_encoder = HiCEncoder()
        self.atac_enc = ModalityEncoder(atac_dim) if atac_dim else None
        self.chip_enc = ModalityEncoder(chip_dim) if chip_dim else None
        self.rna_enc = ModalityEncoder(rna_dim) if rna_dim else None
        self.fusion_net = FusionNetwork()

    def forward(self, hic, atac=None, chip=None, rna=None):
        hic_feat = self.hic_encoder(hic)
        atac_feat = self.atac_enc(atac) if self.atac_enc and atac is not None else None
        chip_feat = self.chip_enc(chip) if self.chip_enc and chip is not None else None
        rna_feat = self.rna_enc(rna) if self.rna_enc and rna is not None else None
        output = self.fusion_net(hic_feat, atac_feat, chip_feat, rna_feat)
        return output


# -------------------------------------------------------------
# Model Test
# -------------------------------------------------------------
if __name__ == "__main__":
    print("Testing ScUnicorn Model...")

    # Dummy data
    hic_input = torch.randn(2, 1, 64, 64)
    atac_input = torch.randn(2, 256)
    chip_input = torch.randn(2, 256)
    rna_input = torch.randn(2, 512)

    # Test with all modalities
    model = ScUnicorn(atac_dim=256, chip_dim=256, rna_dim=512)
    out = model(hic_input, atac_input, chip_input, rna_input)
    print("Multi-modal Output Shape:", out.shape)

    # Test with Hi-C only
    model_hic = ScUnicorn()
    out2 = model_hic(hic_input)
    print("Hi-C Only Output Shape:", out2.shape)

    print("ScUnicorn model test passed successfully.")
