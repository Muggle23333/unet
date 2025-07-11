import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(UNet1D, self).__init__()
        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool1d(2)
        self.encoder2 = self._block(features, features*2)
        self.pool2 = nn.MaxPool1d(2)
        # self.encoder3 = self._block(features*2, features*4)
        # self.pool3 = nn.MaxPool1d(2)   
        self.bottleneck = self._block(features*2, features*4)


        # 后续同理，连接后续 up3/decoder3 ...
        # self.up3 = nn.ConvTranspose1d(features*8, features*4, kernel_size=2, stride=2)
        # self.decoder3 = self._block(features*8, features*4)
        self.up2 = nn.ConvTranspose1d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features*4, features*2)
        self.up1 = nn.ConvTranspose1d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features*2, features)
        self.final = nn.Conv1d(features, out_channels, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        # enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool2(enc2))

        # dec3 = self.up3(bottleneck)
        # dec3 = torch.cat((dec3, enc3), dim=1)
        # dec3 = self.decoder3(dec3)
        dec2 = self.up2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.final(dec1))