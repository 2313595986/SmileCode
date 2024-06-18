import datetime
import argparse
import time
import logging
import sys
import shutil
from utils.data_preparation import get_train_val_test_balance
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from dataloader.ProstateDataset import *
from dataloader.data_aug import *
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from validation import val
from networks.resnet_AMF import Encoder2_AMF
# from networks.resnet_MFFusion import Encoder2_MFFusion
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
from networks_ab.resnet3d import resnet50


parser = argparse.ArgumentParser()
# parser.add_argument("--exp_name", type=str, default='EDNet_144*144*200_0312/'
#                                                     'MFFusion_Elasto_B_E_gray_aug_fold2')
parser.add_argument("--exp_name", type=str, default='debug')
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--fold", type=str, default='BMode612')
parser.add_argument("--fold_path", type=str, default='utils')
parser.add_argument("--data_path", type=str,
                    default='/hy-tmp/Workspaces/datasets/BModeMaskNPZ200*144*144')
parser.add_argument("--swe_path", type=str,
                    default='/hy-tmp/Workspaces/datasets/swe_preprocess/swe_gray200*144*144')
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
    model = model1(1, 1, 2)
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
    train_dataset_positive = PD1C_B_E_gary(pos_list[0], args.data_path, args.swe_path,
                                     transform=transforms.Compose([
                                         Normalization('volume2'),
                                         RandomRotateTransform(mask_prefix=True, angle_range=(-10, 10),
                                                               p_per_sample=0.2),
                                         MirrorTransform(mask_prefix=True, axes=(-3, -2, -1)),
                                         ToTensor(mask_prefix=True, channels=0)]))
    train_dataset_negative = PD1C_B_E_gary(neg_list[0], args.data_path, args.swe_path,
                                      transform=transforms.Compose([
                                          Normalization('volume2'),
                                          RandomRotateTransform(mask_prefix=True, angle_range=(-10, 10),
                                                                p_per_sample=0.2),
                                          MirrorTransform(mask_prefix=True, axes=(-3, -2, -1)),
                                          ToTensor(mask_prefix=True, channels=0)]))

    val_datasets = PD1C_B_E_gary(pos_list[2] + neg_list[2], args.data_path, args.swe_path,
                            transform=transforms.Compose([
                                            Normalization('volume2'),
                                            ToTensor(mask_prefix=True, channels=0)]))

    def worker_init_fn(worker_id):
        random.seed(args.seed1 + worker_id)

    train_dataloader_positive = DataLoader(train_dataset_positive,
                                           batch_size=args.batchsize_positive,
                                           shuffle=True,
                                           num_workers=args.num_workers,
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
        ce_loss_epoch = 0.0
        i_batch = 0

        train_prefetcher_pos = data_prefetcher(train_dataloader_positive)
        train_prefetcher_neg = data_prefetcher(train_dataloader_negative)
        pos_batch = train_prefetcher_pos.next()
        neg_batch = train_prefetcher_neg.next()

        y_true = []
        y_pred = []
        y_pred_sm = []

        model.train()
        while pos_batch is not None and neg_batch is not None:

            start_time = time.time()

            pos_volume, neg_volume = pos_batch['volume1'], neg_batch['volume1']
            pos_swe, neg_swe = pos_batch['volume2'], neg_batch['volume2']
            pos_label, neg_label = pos_batch['cspca'], neg_batch['cspca']
            train_case = pos_batch['name'] + neg_batch['name']

            train_volume = torch.cat([pos_volume, neg_volume], dim=0).cuda().float()
            train_label = torch.cat([pos_label, neg_label], dim=0).cuda()
            train_swe = torch.cat([pos_swe, neg_swe], dim=0).cuda().float()
            optimizer.zero_grad()

            out = model(train_volume, train_swe)

            out_sm = F.softmax(out, dim=1)
            y_true.extend(train_label)
            y_pred.extend(torch.max(out, 1)[1])
            y_pred_sm.extend(out_sm[:, 1])

            ce_loss = celoss(out, train_label)
            loss = ce_loss

            loss_epoch += loss
            ce_loss_epoch += ce_loss
            loss.backward()
            optimizer.step()

            # write image
            if n_total_iter == 0 and args.write_image:
                if train_volume.size(1) == 1:
                    img1 = train_volume[0, :, :, :, ::10].repeat(3, 1, 1, 1).permute(3, 0, 2, 1)
                else:
                    img1 = train_volume[0, :, :, :, ::10].permute(3, 0, 2, 1)
                grid_image = make_grid(img1, 8, normalize=False)
                writer.add_image('PreviewImage_positive_{}'.format(train_case[0]), grid_image, n_total_iter)

                if train_swe.size(1) == 1:
                    img1 = train_swe[0, :, :, :, ::10].repeat(3, 1, 1, 1).permute(3, 0, 2, 1)
                else:
                    img1 = train_swe[0, :, :, :, ::10].permute(3, 0, 2, 1)
                grid_image = make_grid(img1, 8, normalize=False)
                writer.add_image('PreviewSWE_positive_{}'.format(train_case[0]), grid_image, n_total_iter)

            pos_batch = train_prefetcher_pos.next()
            neg_batch = train_prefetcher_neg.next()
            n_total_iter += 1
            end_time = time.time()
            used_time = datetime.timedelta(seconds=(end_time-start_time)).seconds
            logging.info("[Epoch: %4d/%d] [Train index: %2d/%d] [loss: %f] [used time: %ss]"
                         % (epoch, args.max_epoch, i_batch + 1, len(train_dataloader_negative),
                             loss.item(), used_time))
            # logging.info("case id: {}   label: {}".format(train_case, train_label.cpu()))
            i_batch += 1

        y_true = torch.stack(y_true, dim=0)
        # y_pred = torch.stack(y_pred, dim=0)
        y_pred_sm = torch.stack(y_pred_sm, dim=0)
        # train_acc = (y_pred == y_true).sum() / y_pred.size(0)
        fpr, tpr, thresholds_roc = roc_curve(y_true.cpu().data.numpy(), y_pred_sm.cpu().data.numpy(), pos_label=1)
        train_auc = auc(fpr, tpr)

        writer.add_scalar("Loss/loss", loss_epoch.item()/len(train_dataloader_negative), global_step=epoch)
        writer.add_scalar("Loss/celoss", ce_loss_epoch.item()/len(train_dataloader_negative), global_step=epoch)
        writer.add_scalar("Loss/lr", lr, global_step=epoch)
        writer.add_scalar("train/auc", train_auc, epoch)
        logging.info("epoch: {}  train auc: {:.4f}".format(epoch, train_auc))

        if epoch % args.per_val_epoch == 0:
            best_auc = val(model, val_dataloader, writer, epoch, exp_save_path, best_auc)

        if epoch % args.per_save_model == 0:
            torch.save(model.module.state_dict(), '{}/ckp_model/model_{}.pth'.format(exp_save_path, epoch))

    writer.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train()