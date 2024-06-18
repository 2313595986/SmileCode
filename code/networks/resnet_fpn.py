import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')


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
        # self.SE_block = SEblock(channels * self.expansion)

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


class ResNet503D(nn.Module):
    def __init__(self, num_channels=1, num_classes=2):
        super().__init__()
        layers = [3, 4, 6, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv3d(num_channels, 64, kernel_size=(7, 7, 3), stride=(2, 2, 1), padding=(3, 3, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        self.layer1 = self._make_layer(Bottleneck, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        self.dropout = nn.Dropout(0.4)
        # self.SE_block = SEblock(512 * Bottleneck.expansion)
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

    def forward(self, x, debug=False):
        out = self.conv1(x)
        out = self.bn1(out)
        out_0 = self.relu(out)
        out = self.dropout(out_0)
        out = self.max_pool(out)
        out_1 = self.layer1(out)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        # out = self.SE_block(out_4)
        # out = self.avg_pool(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        return out_1, out_2, out_3, out_4


def init_conv_weights(layer, weights_std=0.01,  bias=0):
    '''
    RetinaNet's layer initialization

    :layer
    :

    '''
    nn.init.xavier_normal(layer.weight)
    nn.init.constant(layer.bias.data, val=bias)
    return layer


def conv1x1x1(in_channels, out_channels, **kwargs):
    '''Return a 1x1 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Conv3d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)

    return layer


def conv3x3x3(in_channels, out_channels, **kwargs):
    '''Return a 3x3 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Conv3d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)

    return layer


class Resnet50_FPN(nn.Module):
    def __init__(self, num_channels=1, num_classes=2):
        super(Resnet50_FPN, self).__init__()
        # applied in a pyramid
        self.resnet = ResNet503D(num_channels=num_channels, num_classes=num_classes)
        self.pyramid_transformation_1 = conv3x3x3(64, 256, padding=1)
        self.pyramid_transformation_2 = conv1x1x1(256, 256)
        self.pyramid_transformation_3 = conv1x1x1(512, 256)
        self.pyramid_transformation_4 = conv1x1x1(1024, 256)
        self.pyramid_transformation_5 = conv1x1x1(2048, 256)

        # applied after upsampling
        self.upsample_transform_1 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_3 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_4 = conv3x3x3(256, 256, padding=1)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256, num_classes)
        self.SE_block = SEblock(256)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(256)
        self.bn2 = nn.BatchNorm3d(256)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm3d(256)
        self.bn6 = nn.BatchNorm3d(256)
        self.bn7 = nn.BatchNorm3d(256)

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        depth, height, width = scaled_feature.size()[2:]
        return F.upsample(original_feature, scale_factor=scale_factor)[:, :, :depth, :height, :width]

    def forward(self, x):
        resnet_feature_2, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)
        pyramid_feature_5 = self.relu(self.bn1(self.pyramid_transformation_5(resnet_feature_5)))  # transform c5 from 2048d to 256d
        pyramid_feature_4 = self.relu(self.bn2(self.pyramid_transformation_4(resnet_feature_4)))     # transform c4 from 1024d to 256d
        upsampled_feature_5 = self._upsample(pyramid_feature_5, pyramid_feature_4)   # deconv c5 to c4.size

        pyramid_feature_4 = self.relu(self.bn3(self.upsample_transform_4(
            torch.add(upsampled_feature_5, pyramid_feature_4)               # add up-c5 and c4, and conv
        )))

        pyramid_feature_3 = self.relu(self.bn4(self.pyramid_transformation_3(resnet_feature_3)))     # transform c3 from 512d to 256d
        upsampled_feature_4 = self._upsample(pyramid_feature_4, pyramid_feature_3)    # deconv c4 to c3.size

        pyramid_feature_3 = self.relu(self.bn5(self.upsample_transform_3(
            torch.add(upsampled_feature_4, pyramid_feature_3)               # add up-c4 and c3, and conv
        )))

        pyramid_feature_2 = self.relu(self.bn6(self.pyramid_transformation_2(resnet_feature_2)))                              # c2 is 256d, so no need to transform
        upsampled_feature_3 = self._upsample(pyramid_feature_3, pyramid_feature_2)    # deconv c3 to c2.size

        pyramid_feature_2 = self.relu(self.bn7(self.upsample_transform_2(
            torch.add(upsampled_feature_3, pyramid_feature_2)              # add up-c3 and c2, and conv
        )))

        pyramid_feature_1 = self.SE_block(pyramid_feature_2)
        out = self.avg(pyramid_feature_1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    import os
    import torch.nn.functional as F
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    model = Resnet50_FPN(num_channels=1, num_classes=2)
    model = nn.DataParallel(model).cuda()
    inputs = torch.randn([2, 1, 144, 144, 200]).cuda()
    out = model(inputs)
    out = F.softmax(out, 1)
    print(out)
