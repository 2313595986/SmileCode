import math
import torch
import torch.nn as nn
import torch.hub
import torch.nn.functional as F
import os
import numpy as np
import warnings
from skimage import transform
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
    expansion = 2

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


class Encoder2_MFFusion(nn.Module):

    def __init__(self, num_channels1=2, num_classes=2):
        super(Encoder2_MFFusion, self).__init__()
        layers1 = [3, 4, 6, 3]
        self.in_channels = 64
        self.conv1_1 = nn.Conv3d(num_channels1, 64, kernel_size=(7, 7, 3), stride=(2, 2, 1), padding=(3, 3, 1), bias=False)
        self.bn1_1 = nn.InstanceNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        self.layer1_1 = self._make_layer(Bottleneck, 64, layers1[0], stride=1)
        self.layer1_2 = self._make_layer(Bottleneck, 128, layers1[1], stride=2)
        self.layer1_3 = self._make_layer(Bottleneck, 256, layers1[2], stride=2)
        self.layer1_4 = self._make_layer(Bottleneck, 512, layers1[3], stride=2)

        self.FF0 = FeatureFusion(64, 72, 72, 200, 16, 16, 16)
        self.FF1 = FeatureFusion(128, 36, 36, 200, 8, 8, 16)
        self.FF2 = FeatureFusion(256, 18, 18, 100, 4, 4, 8)
        self.FF3 = FeatureFusion(512, 9, 9, 50, 2, 2, 4)
        self.FF4 = FeatureFusion(1024, 5, 5, 25, 2, 2, 4)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc_fusion = nn.Linear(1024, num_classes)
        self.fc_fusion_refine = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.sig = nn.Softmax()

    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def get_model_py_path(self):
        return os.path.abspath(__file__)

    def forward(self, x1, x2):
        bs, _, _, _, _ = x1.size()
        x = torch.cat((x1, x2), dim=1)

        layer1_0_out = self.relu(self.bn1_1(self.conv1_1(x)))
        fusion_out0 = self.FF0(layer1_0_out)

        layer1_1_in = self.max_pool(fusion_out0)
        layer1_1_out = self.layer1_1(layer1_1_in)
        fusion_out1 = self.FF1(layer1_1_out)

        layer1_2_out = self.layer1_2(fusion_out1)
        fusion_out2 = self.FF2(layer1_2_out)

        layer1_3_out = self.layer1_3(fusion_out2)
        fusion_out3 = self.FF3(layer1_3_out)

        layer1_4_out = self.layer1_4(fusion_out3)
        fusion_out4 = self.FF4(layer1_4_out)

        out_fusion = self.avg_pool(fusion_out4).view(bs, -1)
        out_cls_fusion = self.fc_fusion(out_fusion)

        return out_cls_fusion


class SpatialFusion(nn.Module):
    def __init__(SpatialFusion, in_channels):
        super(CrossAtt, self).__init__()
        self.conv1 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, stride=1)
        self.bn1 = nn.InstanceNorm3d(in_channels)
        self.bn2 = nn.InstanceNorm3d(in_channels)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()

    def forward(self, f1, f2):

        avg_feature = f1 * f2
        attention_map1 = self.sigmod(self.bn1(self.conv1(torch.cat((f1, avg_feature), dim=1))))
        attention_map2 = self.sigmod(self.bn2(self.conv2(torch.cat((f2, avg_feature), dim=1))))

        attention_map_sm1 = torch.exp(attention_map1) / (torch.exp(attention_map1) + torch.exp(attention_map2))
        attention_map_sm2 = torch.exp(attention_map2) / (torch.exp(attention_map1) + torch.exp(attention_map2))

        attention_feature1 = f1 * attention_map_sm1
        attention_feature2 = f2 * attention_map_sm2
        fusion_attention_feature = attention_feature1 + attention_feature2
        return fusion_attention_feature


class DimensionAttention(nn.Module):
    def __init__(self, in_channels, x_s, y_s, z_s, r1, r2, r3):
        super(DimensionAttention, self).__init__()
        self.sex = SEblock(x_s, r1)
        self.sey = SEblock(y_s, r2)
        self.sez = SEblock(z_s, r3)

    def forward(self, f1):
        f1_x = self.sex(f1.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        f1_y = self.sey(f1.permute(0, 3, 2, 1, 4)).permute(0, 3, 2, 1, 4)
        f1_z = self.sez(f1.permute(0, 4, 2, 3, 1)).permute(0, 4, 2, 3, 1)
        out = (f1_x + f1_y + f1_z) / 3
        return out


class FeatureFusion(nn.Module):
    def __init__(self, in_channels, x_s, y_s, z_s, r_x, r_y, r_z):
        super(FeatureFusion, self).__init__()
        self.sa = DimensionAttention(in_channels, x_s, y_s, z_s, r_x, r_y, r_z)
        self.ca = SpatialFusion(in_channels)
        self.conv = nn.Conv3d(in_channels*2, in_channels, kernel_size=1, stride=1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, f1, f2):
        f_sa = self.sa(f1)
        f_ca = self.ca(f1, f2)
        out = torch.cat((f_sa, f_ca), dim=1)
        out = self.conv(out)
        out = self.norm(out)
        out = self.relu(out)
        return out, f_ca


if __name__ == '__main__':
    import os
    import torch.nn.functional as F

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = Encoder2_MFFusion(2, num_classes=2).cuda()
    # model = nn.DataParallel(model).cuda()
    model.train()
    input1 = torch.randn([2, 1, 144, 144, 200]).cuda()
    out_cls = model(input1, input1)
    out = F.softmax(out_cls, 1)
    print(out.size())
    # grad_cam = GradCAM(model, [144, 144, 200])
    # cam10 = grad_cam.forward(input1[0:1], input1[0:1], label=np.zeros(1))
    # cam11 = grad_cam.forward(input1[0:1], input1[0:1], label=np.ones(1))

    # print(cam10.size())
