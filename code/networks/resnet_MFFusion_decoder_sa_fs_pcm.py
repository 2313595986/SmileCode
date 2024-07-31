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


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)):
        super(Upsample, self).__init__()
        self.ConvTrans = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

    def forward(self, x):
        return self.norm(self.ConvTrans(x))


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class ConvDropoutNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), is_dropout=False):
        super(ConvDropoutNormReLU, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=[(i - 1) // 2 for i in kernel_size])
        self.norm = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        if is_dropout:
            self.dropout = nn.Dropout3d(p=0.2, inplace=True)
        else:
            self.dropout = Identity()

        self.all = nn.Sequential(self.conv, self.dropout, self.norm, self.nonlin)

    def forward(self, x):
        return self.all(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 conv_stride=(1, 1, 1), is_dropout=False):
        super(DecoderBlock, self).__init__()
        self.conv = ConvDropoutNormReLU(in_channels, out_channels, kernel_size, conv_stride, is_dropout)

    def forward(self, x):
        return self.conv(x)


class Encoder2_MFFusion_Decoder(nn.Module):

    def __init__(self, num_channels1=1, num_channels2=1, num_classes=2):
        super(Encoder2_MFFusion_Decoder, self).__init__()
        layers1 = [3, 4, 6, 3]
        layers2 = [2, 2, 2, 2]
        self.in_channels = 64
        self.conv1_1 = nn.Conv3d(num_channels1, 64, kernel_size=(7, 7, 3), stride=(2, 2, 1), padding=(3, 3, 1), bias=False)
        self.bn1_1 = nn.InstanceNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))

        self.layer1_1 = self._make_layer(Bottleneck, 64, layers1[0], stride=1)
        self.layer1_2 = self._make_layer(Bottleneck, 128, layers1[1], stride=2)
        self.layer1_3 = self._make_layer(Bottleneck, 256, layers1[2], stride=2)
        self.layer1_4 = self._make_layer(Bottleneck, 512, layers1[3], stride=2)

        self.in_channels = 64
        self.conv2_1 = nn.Conv3d(num_channels2, 64, kernel_size=(7, 7, 3), stride=(2, 2, 1), padding=(3, 3, 1), bias=False)
        self.bn2_1 = nn.InstanceNorm3d(64)
        self.layer2_1 = self._make_layer(Bottleneck, 64, layers2[0], stride=1)
        self.layer2_2 = self._make_layer(Bottleneck, 128, layers2[1], stride=2)
        self.layer2_3 = self._make_layer(Bottleneck, 256, layers2[2], stride=2)
        self.layer2_4 = self._make_layer(Bottleneck, 512, layers2[3], stride=2)

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

        self.up1 = Upsample(1024, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up2 = Upsample(512, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up3 = Upsample(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up4 = Upsample(128, 64, kernel_size=(2, 2, 1), stride=(2, 2, 1))

        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        self.out_conv = nn.Conv3d(64, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)

        self.conv_sa1 = nn.Conv3d(1024, 1024, kernel_size=1, stride=1)
        self.conv_sa2 = nn.Conv3d(1024, 1024, kernel_size=1, stride=1)
        self.norm_sa1 = nn.InstanceNorm3d(1024)
        self.norm_sa2 = nn.InstanceNorm3d(1024)
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

    def forward(self, x1, x2, mask=None, fg_prototype=None, bg_prototype=None, stage='train'):
        bs, _, _, _, _ = x1.size()

        layer1_0_out = self.relu(self.bn1_1(self.conv1_1(x1)))
        layer2_0_out = self.relu(self.bn2_1(self.conv2_1(x2)))
        fusion_out0, fusion_ca0 = self.FF0(layer1_0_out, layer2_0_out)

        layer1_1_in = self.max_pool(layer1_0_out + fusion_out0)
        layer2_1_in = self.max_pool(layer2_0_out + fusion_ca0)

        layer1_1_out = self.layer1_1(layer1_1_in)
        layer2_1_out = self.layer2_1(layer2_1_in)
        fusion_out1, fusion_ca1 = self.FF1(layer1_1_out, layer2_1_out)

        layer1_2_in = layer1_1_out + fusion_out1
        layer2_2_in = layer2_1_out + fusion_ca1
        layer1_2_out = self.layer1_2(layer1_2_in)
        layer2_2_out = self.layer2_2(layer2_2_in)
        fusion_out2, fusion_ca2 = self.FF2(layer1_2_out, layer2_2_out)

        layer1_3_in = layer1_2_out + fusion_out2
        layer2_3_in = layer2_2_out + fusion_ca2
        layer1_3_out = self.layer1_3(layer1_3_in)
        layer2_3_out = self.layer2_3(layer2_3_in)
        fusion_out3, fusion_ca3 = self.FF3(layer1_3_out, layer2_3_out)

        layer1_4_in = layer1_3_out + fusion_out3
        layer2_4_in = layer2_3_out + fusion_ca3
        layer1_4_out = self.layer1_4(layer1_4_in)
        layer2_4_out = self.layer2_4(layer2_4_in)
        fusion_out4, _ = self.FF4(layer1_4_out, layer2_4_out)

        if fg_prototype is not None:
            # 1*1024*5*5*25
            pos_fusion4 = fusion_out4[0:1]
            # 2*1024*1*1*1
            prototype = torch.cat((bg_prototype, fg_prototype), dim=0)
            # 1*1**5*5*25
            pos_bg_cos = F.cosine_similarity(pos_fusion4.view(1, 1024, -1),
                                             bg_prototype.squeeze(-1).squeeze(-1)).view(1, 5, 5, 25).unsqueeze(1)
            # 1*1**5*5*25
            pos_fg_cos = F.cosine_similarity(pos_fusion4.view(1, 1024, -1),
                                             fg_prototype.squeeze(-1).squeeze(-1)).view(1, 5, 5, 25).unsqueeze(1)
            # 1*2*5*5*25
            cos_similarity_map_t1 = torch.cat((pos_bg_cos, pos_fg_cos), dim=1)
            sim_map_t1_softamx = cos_similarity_map_t1.softmax(1)

            q_bg_prototype = F.avg_pool3d(((sim_map_t1_softamx[:, 0:1]) * pos_fusion4), pos_fusion4.size()[-3:])
            q_fg_prototype = F.avg_pool3d((sim_map_t1_softamx[:, 1:2] * pos_fusion4), pos_fusion4.size()[-3:])
            # 2*1024*1*1*1
            q_prototype = torch.cat((q_bg_prototype, q_fg_prototype), dim=0)

            fused_prototype1 = (prototype + q_prototype) / 2

            pos_bg_cos_t = F.cosine_similarity(pos_fusion4.view(1, 1024, -1),
                                               fused_prototype1[0:1].squeeze(-1).squeeze(-1)).view(1, 5, 5,
                                                                                                   25).unsqueeze(1)

            pos_fg_cos_t = F.cosine_similarity(pos_fusion4.view(1, 1024, -1),
                                               fused_prototype1[1:2].squeeze(-1).squeeze(-1)).view(1, 5, 5,
                                                                                                   25).unsqueeze(1)
            # 1*2*5*5*25
            cos_similarity_map_t2 = torch.cat((pos_bg_cos_t, pos_fg_cos_t), dim=1)
            cos_similarity_map = (cos_similarity_map_t1 + cos_similarity_map_t2) / 2
            cos_similarity_map_t_softmax = cos_similarity_map.softmax(1)

            # 1*1024*1*1*1
            q_bg_prototype_t = F.avg_pool3d((cos_similarity_map_t_softmax[:, 0:1] * pos_fusion4),
                                            pos_fusion4.size()[-3:])
            q_fg_prototype_t = F.avg_pool3d((cos_similarity_map_t_softmax[:, 1:2] * pos_fusion4),
                                            pos_fusion4.size()[-3:])
            # 2*1024*1*1*1
            q_prototype_t = torch.cat((q_bg_prototype_t, q_fg_prototype_t), dim=0)
            prototype_refine = (prototype + q_prototype_t) / 2

            # 1*1024*1*1*1
            bg_prototype_refine = prototype_refine[0:1]
            fg_prototype_refine = prototype_refine[1:2]

            pos_bg_cos_refine = F.interpolate(F.cosine_similarity(pos_fusion4.view(1, 1024, -1),
                                                                  bg_prototype_refine.squeeze(-1).squeeze(-1)).view(
                1, 5, 5, 25).unsqueeze(1),
                                              [72, 72, 200], mode='trilinear')

            pos_fg_cos_refine = F.interpolate(F.cosine_similarity(pos_fusion4.view(1, 1024, -1),
                                                                  fg_prototype_refine.squeeze(-1).squeeze(-1)).view(
                1, 5, 5, 25).unsqueeze(1),
                                              [72, 72, 200], mode='trilinear')

        out_fusion = self.avg_pool(fusion_out4).view(bs, -1)
        out_cls_fusion = self.fc_fusion(out_fusion)

        out_fusion_up = self.up1(fusion_out4)
        if out_fusion_up.size() != fusion_out3.size():
            out_fusion_up = F.interpolate(out_fusion_up, size=[fusion_out3.size(2), fusion_out3.size(3), fusion_out3.size(4)], mode='trilinear')
        out4_out5 = torch.cat((fusion_out3, out_fusion_up), dim=1)
        y1 = self.decoder1(out4_out5)

        y1_up = self.up2(y1)
        x3_y1 = torch.cat((fusion_out2, y1_up), dim=1)
        y2 = self.decoder2(x3_y1)

        y2_up = self.up3(y2)
        x2_y2 = torch.cat((fusion_out1, y2_up), dim=1)
        y3 = self.decoder3(x2_y2)

        y3_up = self.up4(y3)
        x1_y3 = torch.cat((fusion_out0, y3_up), dim=1)
        y4 = self.decoder4(x1_y3)

        out_seg = self.out_conv(y4)

        if stage == 'train':

            if mask is not None:
                mask[mask == 2] = 0
                mask = F.interpolate(mask.float(), [5, 5, 25], mode='nearest')
                fg_p = fusion_out4[0:mask.size(0)] * mask / torch.sum(mask)
                bg_p = fusion_out4[0:mask.size(0)] * (1 - mask) / torch.sum(1 - mask)

                return [out_cls_fusion, out_seg, fg_p, bg_p]
            else:
                return [out_cls_fusion, out_seg, pos_fg_cos_refine, pos_bg_cos_refine]
        else:
            return [out_cls_fusion, out_seg]


class SpatialFusion(nn.Module):
    def __init__(self, in_channels):
        super(SpatialFusion, self).__init__()
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
        self.se = SEblock(in_channels)

    def forward(self, f1):
        f1_x = self.sex(f1.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        f1_y = self.sey(f1.permute(0, 3, 2, 1, 4)).permute(0, 3, 2, 1, 4)
        f1_z = self.sez(f1.permute(0, 4, 2, 3, 1)).permute(0, 4, 2, 3, 1)
        out = self.se(f1 + f1_x + f1_y + f1_z)
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    model = Encoder2_MFFusion_Decoder(num_channels1=1, num_channels2=1, num_classes=2).cuda()
    # model = nn.DataParallel(model).cuda()
    model.train()
    input1 = torch.randn([2, 1, 144, 144, 200]).cuda()
    out_cls, seg = model(input1, input1, stage='val')
    out = F.softmax(out_cls, 1)
    print(out.size())
    # grad_cam = GradCAM(model, [144, 144, 200])
    # cam10 = grad_cam.forward(input1[0:1], input1[0:1], label=np.zeros(1))
    # cam11 = grad_cam.forward(input1[0:1], input1[0:1], label=np.ones(1))

    # print(cam10.size())
    print(seg.size())
