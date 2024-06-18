import sys
import os
import torch
import numpy as np
from skimage import transform
import SimpleITK as sitk
import argparse
sys.path.append('../')
from dataloader.ProstateDataset import *
from torch.utils.data import DataLoader
from networks1.resnet_MFFusion_decoder_fs_pr import Encoder2_MFFusion_Decoder as model1
from dataloader.data_aug import *
import cv2
from utils.data_preparation import get_train_val_test_balance
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--exp_name', type=str, default='ResNet_144*144*200_Elasto_E_aug_fold1')
parser.add_argument("--data_path", type=str,
                    default='/hy-tmp/Workspaces/datasets/BModeMaskNPZ200*144*144')
parser.add_argument("--swe_path", type=str,
                    default='/hy-tmp/Workspaces/datasets/swe_preprocess/swe_gray200*144*144')
parser.add_argument('--save_cam_path', type=str, default='../inference')
parser.add_argument('--fold', type=str, default='BMode612')
parser.add_argument('--infer_dir', type=str, default='result3d')
args = parser.parse_args()


class GradCAM:
    def __init__(self, model: torch.nn.Module, cam_size):
        self.model = model
        self.model.eval()
        getattr(self.model, 'FF3').register_forward_hook(self.__forward_hook)
        getattr(self.model, 'FF3').register_backward_hook(self.__backward_hook)

        self.num_cls = 2
        self.size = cam_size
        # self.size = [512, 490, 200]
        self.grads = []
        self.fmaps = []

    def __backward_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0].detach())

    def __forward_hook(self, module, input, output):
        self.fmaps.append(output[0])

    def __compute_loss(self, logit, index):
        BCE = torch.nn.CrossEntropyLoss()
        label = torch.LongTensor(index).to(logit.device)
        loss = BCE(logit, label)
        return loss

    def forward(self, img_arr1, img_arr2, label):
        img_input1 = torch.unsqueeze(img_arr1, axis=0).cuda().float()
        img_input2 = torch.unsqueeze(img_arr2, axis=0).cuda().float()


        # forward
        y = self.model(img_input1, img_input2, stage='val')
        if isinstance(y, list):
            output = y[0]

        # backward
        self.model.zero_grad()
        loss = self.__compute_loss(output, label)
        loss.backward()

        # generate CAM
        grads_val = self.grads[0].cpu().data.numpy().squeeze()
        fmap = self.fmaps[0].cpu().data.numpy().squeeze()
        cam = self.__compute_cam(fmap, grads_val)

        self.fmaps.clear()
        self.grads.clear()
        return cam

    def __compute_cam(self, feature_map, grads):

        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        alpha = np.mean(grads, axis=(1, 2, 3))  # GAP
        for k, ak in enumerate(alpha):
            cam += ak * feature_map[k]  # linear combination
        cam = np.maximum(cam, 0)  # relu
        cam = transform.resize(cam, self.size, order=3, preserve_range=True)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        return cam


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    save_path = os.path.join(args.save_cam_path, args.exp_name, 'fold612', args.infer_dir)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path, exist_ok=True)

    net = nn.DataParallel(model1(1, 1, 2)).cuda()
    net.load_state_dict(torch.load('../model/EDNet_144*144*200_'
                                   '0229_miccai24/Decoder2_sa_box4_Elasto_B_E_gray_aug_fold2_camloss_'
                                   'weight0.001_fs2_refine_sm0.86/ckp_model/model_best.pth'))

    net = net.module
    case_id_list = ['003', '005', '021', '041', '049', '050', '054', '057', '062', '067', '070', '079',
                    '086', '090', '096', '098', '119', '130']
    # z_num_list = [1105, 571, 954, 813, 822, 507, 514, 574, 662, 682, 708, 483,
    #               394, 640, 935, 715, 779, 589, 686, 806]
    # pos_list, neg_list = get_train_val_test_balance('utils', args.fold)
    # case_id_list = pos_list[0] + pos_list[1] + pos_list[2]
    cam_dataset = PD1C_B_E_gary(case_id_list, args.data_path, args.swe_path,
                                      transform=transforms.Compose([
                                          # RGBTOGRAY(),
                                          Normalization('volume2'),
                                          # GaussianBlur(),
                                          ToTensor(mask_prefix=True, channels=0)]))
    cam_dataloader = DataLoader(cam_dataset, batch_size=None, shuffle=False)
    for i_batch, batch in enumerate(cam_dataloader):
        sample1 = batch['volume1']
        sample2 = batch['volume2']
        label = batch['cspca']
        case_id = batch['name']
        print("case id: {}   label: {}".format(case_id, label))
        grad_cam = GradCAM(net, [144, 144, 200])
        cam = grad_cam.forward(sample1, sample2, label=np.ones(1))
        # c x y z -> z y x
        sample = sample1[0, ...].numpy().transpose(2, 1, 0)
        # sample = F.interpolate(sample1.unsqueeze(0), (144, 144, 200)).squeeze(0).numpy().transpose(3, 2, 1, 0) * 255
        # sample = sample[..., 0].numpy().transpose(2, 1, 0)
        cam = cam.transpose(2, 1, 0) * 255
        sitk.WriteImage(sitk.GetImageFromArray(cam), '{}.nii.gz'.format(case_id))
        # if np.sum(cam) == 0:
        #     print(np.sum(cam))
        #     continue
        # cam = np.expand_dims(cam, axis=3).repeat(3, -1)
        # save_case_path = os.path.join(save_path, case_id)
        # os.makedirs(save_case_path, exist_ok=True)
        # for z_i in range(200):
        #     # print(np.sum(cam[z_i]))
        #     rgb_cam = cv2.cvtColor(cam[z_i], cv2.COLOR_BGR2RGB)
        #     rgb_cam = cv2.applyColorMap(rgb_cam.astype(np.uint8), cv2.COLORMAP_JET)
        #     # # sample_cam = sample[z_i, ...] + rgb_cam
        #     alpha = 0.7  # 可以调整透明度
        #     sample_rgb = cv2.cvtColor(sample[z_i, ...], cv2.COLOR_GRAY2BGR)
        #     overlay_image = cv2.addWeighted(sample_rgb.astype(np.uint8), alpha, rgb_cam, 1 - alpha, 0)
        #     cv2.imwrite('{}/{}.jpg'.format(save_case_path, z_i), rgb_cam)
        # plt.imshow(cam)
        # plt.axis('off')
        # plt.show()
        # cam_sample = sitk.GetImageFromArray(cam*255 + sample*255)
        # sitk.WriteImage(cam_sample, os.path.join('{}/{}_cam_img.nii.gz'.format(save_path, case_id)))
        # cam = (cam * 255)
        # cam = sitk.GetImageFromArray(cam)
        # sitk.WriteImage(cam, os.path.join('{}/{}_cam.nii.gz'.format(save_path, case_id)))
        #
        # sample = sitk.GetImageFromArray(sample)
        # sitk.WriteImage(sample, os.path.join('{}/{}_img.nii.gz'.format(save_path, case_id)))

