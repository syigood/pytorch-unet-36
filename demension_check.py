


## 네트워크를 구성하고 dimension을 제대로 나오는지 확인할때 사용

model = UNet()
model.CBR2d(1, 10)
##

# input dimension : [batch size, 3, 32, 32]

def dimension_check():
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

def dimension_check():
    up_conv4 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                  kernel_size=2, stride=2, padding=0, bias=True)
        )

    x = torch.randn(5, 512, 28, 28) # torch.randn(차원) 정의한 차원으로 데이터 랜덤 생성
    y = up_conv4(x)

    print(y.shape)

dimension_chech()
##
