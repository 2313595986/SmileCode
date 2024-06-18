# ResNet3D as presented in Video Classification with Channel-Separated Convolutional Networks(https://arxiv.org/pdf/1904.02811v4.pdf)

import torch
import torch.nn as nn


class SEblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        y = x * y.expand(x.size())
        return y


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.SE_block = SEblock(channels * self.expansion)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        self.conv3 = nn.Conv3d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channels * self.expansion)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels * self.expansion)
            )

    def forward(self, x):
        shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)
        # out = self.SE_block(out)

        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_channels=1, num_classes=2):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv3d(num_channels, 64, kernel_size=(7, 7, 3), stride=(2, 2, 1), padding=(3, 3, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(0.4)
        self.SE_block = SEblock(512 * block.expansion)
        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out_0 = self.relu(out)
        out = self.dropout(out_0)
        out = self.max_pool(out)
        out_1 = self.layer1(out)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        out = out_4
        out = self.SE_block(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



def resnet26(num_channels, num_classes):
    return ResNet3D(Bottleneck, [1, 2, 4, 1], num_channels=num_channels, num_classes=num_classes)


def resnet50(num_channels, num_classes):
    return ResNet3D(Bottleneck, [3, 4, 6, 3], num_channels=num_channels, num_classes=num_classes)


def resnet101(num_classes):
    return ResNet3D(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes):
    return ResNet3D(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


if __name__ == '__main__':
    import os
    import torch.nn.functional as F
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model = resnet50(num_channels=1, num_classes=2)
    model = nn.DataParallel(model).cuda()
    inputs = torch.randn([1, 1, 100, 224, 224]).cuda()
    out = model(inputs)
    out = F.softmax(out, 1)
    print(out)
