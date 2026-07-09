import torch
import torch.nn as nn

class UNet1D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet1D, self).__init__()

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool1d(2)

        self.bottleneck = self.conv_block(512, 1024)

        self.upconv4 = self.upconv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        self.conv_out = nn.Conv1d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        if dec4.shape[2] != enc4.shape[2]:
            dec4 = nn.functional.pad(dec4, (0, enc4.shape[2] - dec4.shape[2]))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        if dec3.shape[2] != enc3.shape[2]:
            dec3 = nn.functional.pad(dec3, (0, enc3.shape[2] - dec3.shape[2]))
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        if dec2.shape[2] != enc2.shape[2]:
            dec2 = nn.functional.pad(dec2, (0, enc2.shape[2] - dec2.shape[2]))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.shape[2] != enc1.shape[2]:
            dec1 = nn.functional.pad(dec1, (0, enc1.shape[2] - dec1.shape[2]))
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        x = self.conv_out(dec1)
        x = torch.mean(x, dim=2)

        if torch.any(torch.isnan(x)):
            raise ValueError

        return x

# ========================= EOF ====================================================================