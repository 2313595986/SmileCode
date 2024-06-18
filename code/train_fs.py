import datetime
import argparse
import time
import logging
import sys
import shutil

import numpy as np
import torch
from skimage.filters import threshold_otsu
from utils.data_preparation import get_train_val_test_balance
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from dataloader.ProstateDataset import *
from dataloader.data_aug import *
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import make_grid
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from validation import val_mask
from dataloader.losses import LogBarrierLoss, BCE
from networks.resnet_MFFusion_decoder_sa_fs_pr import Encoder2_MFFusion_Decoder
from utils.utils import standardized_seg


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default='EDNet_144*144*200_0229/'
                                                    'Decoder2_sa_box4_Elasto_B_E_gray_aug_fold2_camloss_weight0.001_fs2_refine_sm')
# parser.add_argument("--exp_name", type=str, default='debug'
parser.add_argument("--gpu", type=str, default='2')
parser.add_argument("--fold", type=str, default='BMode612')
parser.add_argument("--pretrain", type=float, default=0)
parser.add_argument("--fold_path", type=str, default='utils')
parser.add_argument("--data_path", type=str,
                    default='/hy-tmp/Workspaces/datasets/BModeMaskNPZ200*144*144')
parser.add_argument("--swe_path", type=str,
                    default='/hy-tmp/Workspaces/datasets/swe_preprocess/swe_gray200*144*144')
parser.add_argument("--box_path", type=str,
                    default='/hy-tmp/Workspaces/datasets/TumorMaskNPY200*72*72')
parser.add_argument("--write_image", type=bool, default=True)
parser.add_argument('--num_channels2', type=int, default=1)
parser.add_argument("--save_root_path", type=str, default='../model')
parser.add_argument("--max_epoch", type=int, default=200)
parser.add_argument("--batchsize_positive", type=int, default=1)
parser.add_argument("--batchsize_negative", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--val_batchsize", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--seed1", type=int, default=1997)
parser.add_argument("--per_val_epoch", type=int, default=2)
parser.add_argument("--per_save_model", type=int, default=5)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--rm_exp", type=bool, default=False)
args = parser.parse_args()


def create_model():
    model = Encoder2_MFFusion_Decoder(num_channels1=1, num_channels2=args.num_channels2, num_classes=args.num_classes)
    model = nn.DataParallel(model)
    return model.cuda()


def save_parameter(exp_save_path, d=True):
    delete = True if os.path.basename(exp_save_path) == 'debug' else d
    if os.path.exists(exp_save_path) is True:
        assert delete is True
        shutil.rmtree(exp_save_path)
    os.makedirs(exp_save_path)
    os.makedirs(os.path.join(exp_save_path, 'ckp_model'))

    # save this .py
    py_path_old = sys.argv[0]
    py_path_new = os.path.join(exp_save_path, os.path.basename(py_path_old))
    shutil.copy(py_path_old, py_path_new)
    logging.basicConfig(filename=os.path.join(exp_save_path, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info('PID: {}'.format(os.getpid()))


def reproduce(seed1):
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(seed1)
    np.random.seed(seed1)
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed1)


def train():
    reproduce(args.seed1)
    exp_time = time.localtime()
    exp_time_format = time.strftime("%m-%d-%H-%M", exp_time)
    exp_save_path = os.path.join(args.save_root_path, '{}'.format(args.exp_name))
    save_parameter(exp_save_path, args.rm_exp)
    writer = SummaryWriter(log_dir=exp_save_path)

    print('-------------------------------------- setting --------------------------------------')
    print("experiment name: {}".format(os.path.basename(exp_save_path)))
    print("time: ", exp_time_format)
    print('data name: {}'.format(os.path.basename(args.data_path)))
    print("fold: {}".format(args.fold))
    print("gpu: {}".format(args.gpu))
    print("save path: {}".format(exp_save_path))
    print('-------------------------------------- setting --------------------------------------')

    # load data
    pos_list, neg_list = get_train_val_test_balance(args.fold_path, args.fold)
    # '381', '399', '428', '482'
    box_list = ['381', '482', '428', '399']
    train_pos_list = []
    for c_ in pos_list[0]:
        if c_ not in box_list:
            train_pos_list.append(c_)

    train_dataset_positive_nobox = PD1C_B_E_gary(train_pos_list, args.data_path, args.swe_path,
                                                 transform=transforms.Compose([
                                                     Normalization('volume2'),
                                                     RandomRotateTransform(mask_prefix=True, angle_range=(-10, 10),
                                                                           p_per_sample=0.2),
                                                     MirrorTransform(mask_prefix=True, axes=(-3, -2, -1)),
                                                     # GaussianBlur(),
                                                     ToTensor(mask_prefix=True, channels=0)]))

    train_dataset_positive_box = PD1C_B_E_gary_bbox(box_list, args.data_path, args.swe_path, args.box_path,
                                                    transform=transforms.Compose([
                                                        Normalization('volume2'),
                                                        RandomRotateTransform(mask_prefix=True, box_prefix=True,
                                                                              angle_range=(-10, 10),
                                                                              p_per_sample=0.2),
                                                        MirrorTransform(mask_prefix=True, box_prefix=True,
                                                                        axes=(-3, -2, -1)),
                                                        ToTensor(mask_prefix=True, box_prefix=True, channels=0)]))
    train_dataset_negative = PD1C_B_E_gary(neg_list[0], args.data_path, args.swe_path,
                                      transform=transforms.Compose([
                                          Normalization('volume2'),
                                          RandomRotateTransform(mask_prefix=True, angle_range=(-10, 10), p_per_sample=0.2),
                                          MirrorTransform(mask_prefix=True, axes=(-3, -2, -1)),
                                          # GaussianBlur(),
                                          ToTensor(mask_prefix=True, channels=0)]))

    val_datasets = PD1C_B_E_gary(pos_list[2] + neg_list[2], args.data_path, args.swe_path,
                            transform=transforms.Compose([
                                                    # RGBTOGRAY(),
                                                    Normalization('volume2'),
                                                    # GaussianBlur(),
                                                    ToTensor(mask_prefix=True, channels=0)]))

    def worker_init_fn(worker_id):
        random.seed(args.seed1 + worker_id)

    train_dataloader_positive_nobox = DataLoader(train_dataset_positive_nobox,
                                           batch_size=args.batchsize_positive,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           pin_memory=False,
                                           worker_init_fn=worker_init_fn)

    train_dataloader_positive_box = DataLoader(train_dataset_positive_box,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=False,
                                               worker_init_fn=worker_init_fn)

    train_dataloader_negative = DataLoader(train_dataset_negative,
                                           batch_size=args.batchsize_negative,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           pin_memory=False,
                                           worker_init_fn=worker_init_fn)

    val_dataloader = DataLoader(val_datasets,
                                batch_size=args.val_batchsize,
                                shuffle=True)

    model = create_model()

    model_py_path = model.module.get_model_py_path()
    shutil.copy(model_py_path, os.path.join(exp_save_path, os.path.basename(model_py_path)))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    celoss = nn.CrossEntropyLoss()

    def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
        return initial_lr * (1 - epoch / max_epochs) ** exponent

    n_total_iter = 0
    best_auc = 0

    for epoch in range(args.max_epoch):
        lr = poly_lr(epoch, args.max_epoch, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        loss_epoch = 0.0
        box_seg_loss_epoch = 0.0
        neg_seg_loss_epoch = 0.0
        l_pos_epoch = 0.0
        ce_loss_epoch = 0.0
        i_batch = 0

        train_prefetcher_pos_nobox = data_prefetcher(train_dataloader_positive_nobox)
        train_prefetcher_pos_box = data_prefetcher(train_dataloader_positive_box)
        train_prefetcher_neg = data_prefetcher(train_dataloader_negative)
        pos_batch_nobox = train_prefetcher_pos_nobox.next()
        pos_batch_box = train_prefetcher_pos_box.next()
        neg_batch = train_prefetcher_neg.next()

        y_true = []
        y_pred = []
        y_pred_sm = []

        fg_feature_prototype = torch.zeros((1, 1024, 1, 1, 1)).cuda()
        bg_feature_prototype = torch.zeros((1, 1024, 1, 1, 1)).cuda()

        model.train()
        while neg_batch is not None and pos_batch_nobox is not None and i_batch < 100:

            start_time = time.time()
            pos_cam_sampled = False
            if pos_batch_box is not None:
                box_mark = True
            else:
                box_mark = False

            if box_mark:
                pos_volume, neg_volume = pos_batch_box['volume1'], neg_batch['volume1']
                pos_swe, neg_swe = pos_batch_box['volume2'], neg_batch['volume2']
                pos_label, neg_label = pos_batch_box['cspca'], neg_batch['cspca']
                train_case = pos_batch_box['name'] + neg_batch['name']
                pos_box = pos_batch_box['box'].cuda()
                z_sum = torch.sum(torch.sum(pos_box, dim=2), dim=2).squeeze(0).squeeze(0).cpu().data.numpy()
                nozero_index = [i for i in range(len(z_sum)) if z_sum[i] != 0]

            else:
                pos_volume, neg_volume = pos_batch_nobox['volume1'], neg_batch['volume1']
                pos_swe, neg_swe = pos_batch_nobox['volume2'], neg_batch['volume2']
                pos_label, neg_label = pos_batch_nobox['cspca'], neg_batch['cspca']
                train_case = pos_batch_nobox['name'] + neg_batch['name']

            train_volume = torch.cat([pos_volume, neg_volume], dim=0).cuda().float()
            train_label = torch.cat([pos_label, neg_label], dim=0).cuda()
            train_swe = torch.cat([pos_swe, neg_swe], dim=0).cuda().float()
            optimizer.zero_grad()
            if box_mark:
                out_cls, out_seg, fg_p, bg_p = model(train_volume, train_swe, mask=pos_box)
                fg_feature_prototype += fg_p
                bg_feature_prototype += bg_p
            else:
                out_cls, out_seg, pos_fg_cos, pos_bg_cos = model(train_volume, train_swe,
                                                 bg_prototype=bg_feature_prototype/4,
                                                 fg_prototype=fg_feature_prototype/4)

            out_sm = F.softmax(out_cls, dim=1)
            y_true.extend(train_label)
            y_pred.extend(torch.max(out_cls, 1)[1])
            y_pred_sm.extend(out_sm[:, 1])

            # cam positive loss
            l_pos = 0.
            out_seg_sm = out_seg.softmax(1)
            if epoch >= args.pretrain:
                pos_seg = out_seg[0:1]
                if box_mark:
                    # box supervise
                    box_mask = [pos_box != 2]
                    pos_seg_ce = torch.stack([pos_seg[:, 0:1][box_mask], pos_seg[:, 1:2][box_mask]], dim=1)
                    box_seg_loss = celoss(pos_seg_ce, pos_box[box_mask].long())
                    l_box = box_seg_loss
                    box_seg_loss_epoch += box_seg_loss

                    # seg negative loss
                    seg_neg = out_seg[1:2]
                    seg_gt = torch.zeros_like(seg_neg).cuda()
                    l_neg = celoss(seg_neg, seg_gt[:, 0].long())
                    neg_seg_loss_epoch += l_neg
                    l_weak = l_neg + l_box
                    ce_loss = celoss(out_cls, train_label)
                    loss = ce_loss + l_weak

                    pos_cam_sampled = None
                else:
                    pos_cos = torch.cat((pos_bg_cos, pos_fg_cos), dim=1).softmax(1)[:, 1:2]
                    # pos_cos = pos_fg_cos
                    pos_cos_np = pos_cos.cpu().data.numpy()

                    pos_cos_np_flatten = pos_cos_np.flatten()
                    top5val = np.sort(pos_cos_np_flatten)[-5]
                    bottom5val = np.sort(pos_cos_np_flatten)[5]

                    seed_outs = pos_cos.permute(0, 2, 3, 4, 1).contiguous()
                    fg_seed_mask = [seed_outs[..., 0] >= top5val]
                    bg_seed_mask = [seed_outs[..., 0] <= bottom5val]

                    seed_outs = pos_seg.permute(0, 2, 3, 4, 1).contiguous()
                    fg_seed_outs = seed_outs[fg_seed_mask]
                    bg_seed_outs = seed_outs[bg_seed_mask]

                    if not fg_seed_outs.shape[0] == 0:
                        l_pos_fg = celoss(fg_seed_outs, torch.LongTensor(np.ones(fg_seed_outs.shape[0])).cuda())
                        l_pos = l_pos + l_pos_fg
                        pos_cam_sampled = True
                    if not bg_seed_outs.shape[0] == 0:
                        l_pos_bg = celoss(bg_seed_outs, torch.LongTensor(np.zeros(bg_seed_outs.shape[0])).cuda())
                        l_pos = l_pos + l_pos_bg

                    # seg negative loss
                    seg_neg = out_seg[1:2]
                    seg_gt = torch.zeros_like(seg_neg).cuda()
                    l_neg = celoss(seg_neg, seg_gt[:, 0].long())
                    neg_seg_loss_epoch += l_neg
                    l_pos_epoch += l_pos

                    l_weak = l_neg + l_pos

                    ce_loss = celoss(out_cls, train_label)
                    loss = ce_loss + 0.0001 * l_weak
            else:
                ce_loss = celoss(out_cls, train_label)
                loss = ce_loss

            loss_epoch += loss
            ce_loss_epoch += ce_loss
            loss.backward()
            optimizer.step()

            if box_mark and epoch % 5 == 0:
                img1 = pos_box[0, :, :, :, np.array(nozero_index)].repeat(3, 1, 1, 1).permute(3, 0, 2, 1)
                grid_image = make_grid(img1, 8, normalize=False)
                writer.add_image('{0}/cam_positive_box_gt{0}'.format(pos_batch_box['name'][0]), grid_image,
                                 n_total_iter)

                img1 = out_seg_sm[0, 1:2, :, :, np.array(nozero_index)].repeat(3, 1, 1, 1).permute(3, 0, 2, 1)
                grid_image = make_grid(img1, 8, normalize=False)
                writer.add_image('{0}/positive_seg_pred_box{0}'.format(pos_batch_box['name'][0]), grid_image,
                                     n_total_iter)

            # write image
            if n_total_iter % 100 == 0 and args.write_image:

                if not box_mark:
                    img1 = pos_fg_cos[0, 0:1, :, :, 20:-20:10].repeat(3, 1, 1, 1).permute(3, 0, 2, 1)
                    grid_image = make_grid(img1, 8, normalize=True)
                    writer.add_image('positive_prototype', grid_image, n_total_iter)

                    img1 = out_seg_sm[0, 1:2, :, :, 20:-20:10].repeat(3, 1, 1, 1).permute(3, 0, 2, 1)
                    grid_image = make_grid(img1, 8, normalize=False)
                    writer.add_image('positive_seg_pred', grid_image, n_total_iter)

                    img1 = out_seg_sm[1, 1:2, :, :, 20:-20:20].repeat(3, 1, 1, 1).permute(3, 0, 2, 1)
                    grid_image = make_grid(img1, 8, normalize=False)
                    writer.add_image('negative_seg_pred', grid_image, n_total_iter)

            pos_batch_nobox = train_prefetcher_pos_nobox.next()
            if box_mark:
                pos_batch_box = train_prefetcher_pos_box.next()
            # if pos_batch_box is None:
            #     train_prefetcher_pos_box = data_prefetcher(train_dataloader_positive_box)
            #     pos_batch_box = train_prefetcher_pos_box.next()
            neg_batch = train_prefetcher_neg.next()
            # box_mark = not box_mark
            n_total_iter += 1
            end_time = time.time()
            used_time = datetime.timedelta(seconds=(end_time-start_time)).seconds
            logging.info("[Epoch: %4d/%d] [Train index: %2d/%d] [loss: %.4f] [used time: %ss] [case id: %s] "
                         "[label: %s] [box: %s] [pos cam sampled: %s]"
                         % (epoch, args.max_epoch, i_batch + 1, len(train_dataloader_negative), loss.item(), used_time,
                            str(train_case), str(train_label.cpu()), str(box_mark), str(pos_cam_sampled)))
            # logging.info("case id: {}   label: {}".format(train_case, train_label.cpu()))
            i_batch += 1

        y_true = torch.stack(y_true, dim=0)
        y_pred_sm = torch.stack(y_pred_sm, dim=0)
        fpr, tpr, thresholds_roc = roc_curve(y_true.cpu().data.numpy(), y_pred_sm.cpu().data.numpy(), pos_label=1)
        train_auc = auc(fpr, tpr)

        writer.add_scalar("Loss/loss", loss_epoch.item()/len(train_dataloader_negative), global_step=epoch)
        writer.add_scalar("Loss/celoss", ce_loss_epoch.item()/len(train_dataloader_negative), global_step=epoch)
        writer.add_scalar("Loss/neg_seg_loss", neg_seg_loss_epoch.item()/len(train_dataloader_negative), global_step=epoch)
        writer.add_scalar("Loss/box_seg_loss", box_seg_loss_epoch.item()/(len(train_dataloader_negative)/2), global_step=epoch)
        # writer.add_scalar("Loss/cam_sample_loss", l_pos_epoch.item()/(len(train_dataloader_negative)/2), global_step=epoch)
        writer.add_scalar("Loss/lr", lr, global_step=epoch)
        writer.add_scalar("train/auc", train_auc, epoch)
        logging.info("epoch: {}  train auc: {:.4f}".format(epoch, train_auc))

        if epoch % args.per_val_epoch == 0:
            best_auc = val_mask(model, val_dataloader, writer, epoch, exp_save_path, best_auc)

        if epoch % args.per_save_model == 0:
            torch.save(model.module.state_dict(), '{}/ckp_model/model_{}.pth'.format(exp_save_path, epoch))

    writer.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train()