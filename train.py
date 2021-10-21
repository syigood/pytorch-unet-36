## 라이브러리 추가
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

## 트레이닝 파라미터 설정하기
lr = 1e-3
batch_size = 4
num_epoch = 100

data_dir = './datasets'
ckpt_dir = './checkpoint'
log_dir = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 네트워크 구축하기

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.enc1_1 = self.CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = self.CBR2d(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = self.CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = self.CBR2d(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = self.CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = self.CBR2d(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = self.CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = self.CBR2d(in_channels=512, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = self.CBR2d(in_channels=512, out_channels=1024)
        self.dec5_1 = self.CBR2d(in_channels=1024, out_channels=512)

        self.up_conv4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                           kernel_size=2, stride=2,padding=0, bias=True)

        self.dec4_2 = self.CBR2d(in_channels=2*512, out_channels=512)
        self.dec4_1 = self.CBR2d(in_channels=512, out_channels=256)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                           kernel_size=2, stride=2,padding=0, bias=True)

        self.dec3_2 = self.CBR2d(in_channels=2*256, out_channels=256)
        self.dec3_1 = self.CBR2d(in_channels=256, out_channels=128)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                           kernel_size=2, stride=2,padding=0, bias=True)

        self.dec2_2 = self.CBR2d(in_channels=2*128, out_channels=128)
        self.dec2_1 = self.CBR2d(in_channels=128, out_channels=64)

        self.up_conv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                           kernel_size=2, stride=2,padding=0, bias=True)

        self.dec1_2 = self.CBR2d(in_channels=2*64, out_channels=64)
        self.dec1_1 = self.CBR2d(in_channels=64, out_channels=64)

        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=2,
                                 kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)

        up_conv4 = self.up_conv4(dec5_1)
        cat4 = torch.cat((enc4_2, up_conv4), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        up_conv3 = self.up_conv3(dec4_1)
        cat3 = torch.cat((enc3_2, up_conv3), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        up_conv2 = self.up_conv2(dec3_1)
        cat2 = torch.cat((enc2_2, up_conv2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        up_conv1 = self.up_conv1(dec2_1)
        cat1 = torch.cat((enc1_2, up_conv1), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.conv1x1(dec1_1)

        return x


    def CBR2d(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=bias))
        layers.append(nn.BatchNorm2d(num_features=out_channels))
        layers.append(nn.ReLU())

        print(layers)

        cbr = nn.Sequential(*layers)
        return cbr

##
model = UNet()
model.CBR2d(1, 10)
##

# input dimension : [batch size, 3, 32, 32]

def dimension_chech():
    model = UNet()
    # net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1000, zero_init_residual=False)

    in_layer = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), # --> 16, 64, 16, 16
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #--> # --> 16, 64, 8, 8
        )

    x = torch.randn(16, 3, 32, 32) # torch.randn(차원) 정의한 차원으로 데이터 랜덤 생성
    x = in_layer(x)
    y = model(x)

    print(y.shape)

dimension_chech()

##

def dimension_chech():
    up_conv4 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                  kernel_size=2, stride=2, padding=0, bias=True)
        )

    x = torch.randn(5, 512, 28, 28) # torch.randn(차원) 정의한 차원으로 데이터 랜덤 생성
    y = up_conv4(x)

    print(y.shape)

dimension_chech()
##

