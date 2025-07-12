import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch)
        )
        # 如果 in_ch != out_ch，用1x1卷积升降维
        if in_ch != out_ch:
            self.res_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_conv = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        res = self.res_conv(x)
        return self.relu(out + res)

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet1D, self).__init__()
        features = init_features
        self.encoder1 = ResidualBlock1D(in_channels, features)
        self.pool1 = nn.MaxPool1d(2)
        self.encoder2 = ResidualBlock1D(features, features*2)
        self.pool2 = nn.MaxPool1d(2)
        self.encoder3 = ResidualBlock1D(features*2, features*4)
        self.pool3 = nn.MaxPool1d(2)
        self.bottleneck = ResidualBlock1D(features*4, features*8)

        self.up3 = nn.ConvTranspose1d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock1D(features*8, features*4)
        self.up2 = nn.ConvTranspose1d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock1D(features*4, features*2)
        self.up1 = nn.ConvTranspose1d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock1D(features*2, features)
        self.final = nn.Conv1d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool1(enc2))        
        bottleneck = self.bottleneck(self.pool3(enc3))


        dec3 = self.up3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.final(dec1))