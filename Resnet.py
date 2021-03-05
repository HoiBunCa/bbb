import torch
import torch.nn as nn
import torchvision.models as models


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride1, stride2, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

    def forward(self, x_inp):
        x = self.conv1(x_inp)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            x_downsample = self.downsample(x_inp)
            x_out = torch.add(x, x_downsample)
        else:
            x_out = torch.add(x, x_inp)
        return x_out


def make_layer(BasicBlock, in_channels, out_channels, num_layer, first_layer):
    list_layer = []
    if not first_layer:
        list_layer.append(BasicBlock(in_channels, out_channels, 2, 1, downsample=True))
    else:
        list_layer.append(BasicBlock(in_channels, out_channels, 1, 1, downsample=False))

    for i in range(1, num_layer):
        list_layer.append(BasicBlock(out_channels, out_channels, 1, 1, downsample=False))

    return nn.Sequential(*list_layer)


class ResNet(nn.Module):
    def __init__(self, Block, num_block_each_layer):  # BasicBlock - bottleneckBlock
        # resnet18 [2,2,2,2]
        super(ResNet, self).__init__()
        # Start
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        # layer 1
        self.layer1 = make_layer(Block, 64, 64, num_block_each_layer[0], first_layer=True)
        self.layer2 = make_layer(Block, 64, 128, num_block_each_layer[1], first_layer=False)
        self.layer3 = make_layer(Block, 128, 256, num_block_each_layer[2], first_layer=False)
        self.layer4 = make_layer(Block, 256, 512, num_block_each_layer[3], first_layer=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


x = torch.randn((1, 3, 224, 224))
resnet18_h = ResNet(BasicBlock, [3, 4, 6, 3])
y = resnet18_h(x)
print(y.shape)
