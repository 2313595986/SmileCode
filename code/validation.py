import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
import logging
import torch.nn.functional as F
import numpy as np


def compute_accuracy_by_class(ground_truth, predicted, class_label):
    correct = 0
    total = 0

    for true_label, predicted_label in zip(ground_truth, predicted):
        if true_label == class_label:
            total += 1
            if predicted_label == class_label:
                correct += 1

    if total == 0:
        return 0  # 避免除零错误

    accuracy = correct / total
    return accuracy


def val(model, val_dataloader, writer, epoch, exp_save_path, best_auc):
    model.eval()
    val_y_true = []
    val_y_pred_sm = []

    for val_batch in tqdm(val_dataloader):
        val_volume = val_batch['volume1'].float().cuda()
        val_swe = val_batch['volume2'].float().cuda()
        val_label = val_batch['cspca'].cuda()
        val_name = val_batch['name']
        with torch.no_grad():
            y = model(val_volume, val_swe)
        if isinstance(y, list):
            y = y[0]
        y_sm = F.softmax(y, dim=1)
        val_y_true.extend(val_label)
        val_y_pred_sm.extend(y_sm[:, 1])
        logging.info("case id: {}   label: {}   pred: {}".format(val_name, val_label.cpu(), torch.max(y, 1)[1].cpu()))

    val_y_true = torch.stack(val_y_true, dim=0)
    val_y_pred_sm = torch.stack(val_y_pred_sm, dim=0)
    fpr, tpr, thresholds_roc = roc_curve(val_y_true.cpu().data.numpy(), val_y_pred_sm.cpu().data.numpy(), pos_label=1)
    precision, recall, thresholds = precision_recall_curve(val_y_true.cpu().data.numpy(), val_y_pred_sm.cpu().data.numpy())
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    pred_list = [1 if x > best_threshold else 0 for x in val_y_pred_sm.cpu().data.numpy()]
    acc = accuracy_score(val_y_true.cpu().data.numpy(), pred_list)
    val_auc = auc(fpr, tpr)
    pos_acc = compute_accuracy_by_class(val_y_true, pred_list, 1)
    neg_acc = compute_accuracy_by_class(val_y_true, pred_list, 0)
    logging.info("evaluation result: AUC=%.4f Recall=%.4f Precision=%.4f f1_score=%.4f ACC=%.4f"
                 % (val_auc, recall[best_threshold_index], precision[best_threshold_index], f1_scores[best_threshold_index], acc))
    logging.info('positive acc: {:.4f} negative acc: {:.4f}'.format(pos_acc, neg_acc))
    writer.add_scalar("val/auc", val_auc, global_step=epoch)
    writer.add_scalar("val/recall", recall[best_threshold_index], global_step=epoch)
    writer.add_scalar("val/precision", precision[best_threshold_index], global_step=epoch)
    writer.add_scalar("val/f1", f1_scores[best_threshold_index], global_step=epoch)
    writer.add_scalar("val/acc", acc, global_step=epoch)
    writer.add_scalar("val/pos_acc", pos_acc, global_step=epoch)
    writer.add_scalar("val/neg_acc", neg_acc, global_step=epoch)
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.module.state_dict(), '{}/ckp_model/model_best.pth'.format(exp_save_path))
    writer.add_scalar("val/best_model_result_auc", best_auc, global_step=epoch)

    return best_auc


def val_mask(model, val_dataloader, writer, epoch, exp_save_path, best_auc):
    model.eval()
    val_y_true = []
    val_y_pred_sm = []

    for val_batch in tqdm(val_dataloader):
        val_volume = val_batch['volume1'].float().cuda()
        val_swe = val_batch['volume2'].float().cuda()
        val_mask = val_batch['mask'].float().cuda()
        val_label = val_batch['cspca'].cuda()
        val_name = val_batch['name']
        with torch.no_grad():
            y = model(val_volume, val_swe, stage='val')
        if isinstance(y, list):
            y = y[0]
        y_sm = F.softmax(y, dim=1)
        val_y_true.extend(val_label)
        val_y_pred_sm.extend(y_sm[:, 1])
        logging.info("case id: {}   label: {}   pred: {}".format(val_name, val_label.cpu(), torch.max(y, 1)[1].cpu()))

    val_y_true = torch.stack(val_y_true, dim=0)
    val_y_pred_sm = torch.stack(val_y_pred_sm, dim=0)
    fpr, tpr, thresholds_roc = roc_curve(val_y_true.cpu().data.numpy(), val_y_pred_sm.cpu().data.numpy(), pos_label=1)
    precision, recall, thresholds = precision_recall_curve(val_y_true.cpu().data.numpy(), val_y_pred_sm.cpu().data.numpy())
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    pred_list = [1 if x > best_threshold else 0 for x in val_y_pred_sm.cpu().data.numpy()]
    acc = accuracy_score(val_y_true.cpu().data.numpy(), pred_list)
    val_auc = auc(fpr, tpr)
    pos_acc = compute_accuracy_by_class(val_y_true, pred_list, 1)
    neg_acc = compute_accuracy_by_class(val_y_true, pred_list, 0)
    logging.info("evaluation result: AUC=%.4f Recall=%.4f Precision=%.4f f1_score=%.4f ACC=%.4f"
                 % (val_auc, recall[best_threshold_index], precision[best_threshold_index], f1_scores[best_threshold_index], acc))
    logging.info('positive acc: {:.4f} negative acc: {:.4f}'.format(pos_acc, neg_acc))
    writer.add_scalar("val/auc", val_auc, global_step=epoch)
    writer.add_scalar("val/recall", recall[best_threshold_index], global_step=epoch)
    writer.add_scalar("val/precision", precision[best_threshold_index], global_step=epoch)
    if f1_scores[best_threshold_index] != np.NaN:
        writer.add_scalar("val/f1", f1_scores[best_threshold_index], global_step=epoch)
    writer.add_scalar("val/acc", acc, global_step=epoch)
    writer.add_scalar("val/pos_acc", pos_acc, global_step=epoch)
    writer.add_scalar("val/neg_acc", neg_acc, global_step=epoch)
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.module.state_dict(), '{}/ckp_model/model_best.pth'.format(exp_save_path))
    writer.add_scalar("val/best_model_result_auc", best_auc, global_step=epoch)

    return best_auc


def val_camloss(model, val_dataloader, writer, epoch, exp_save_path, best_auc):
    model.eval()
    val_y_true = []
    # val_y_pred = []
    val_y_pred_sm = []
    with torch.no_grad():
        for val_batch in tqdm(val_dataloader):
            val_volume = val_batch['volume'].float().cuda()
            val_label = val_batch['cspca'].cuda()
            val_name = val_batch['name']
            y, _ = model(val_volume)
            y_sm = F.softmax(y, dim=1)
            val_y_true.extend(val_label)
            # val_y_pred.extend(torch.max(y, 1)[1])
            val_y_pred_sm.extend(y_sm[:, 1])
            logging.info("case id: {}   label: {}   pred: {}".format(val_name, val_label.cpu(), torch.max(y, 1)[1].cpu()))

        val_y_true = torch.stack(val_y_true, dim=0)
        # val_y_pred = torch.stack(val_y_pred, dim=0)
        val_y_pred_sm = torch.stack(val_y_pred_sm, dim=0)
        # val_acc = (val_y_pred == val_y_true).sum() / val_y_true.size(0)
        fpr, tpr, thresholds_roc = roc_curve(val_y_true.cpu().data.numpy(), val_y_pred_sm.cpu().data.numpy(), pos_label=1)
        val_auc = auc(fpr, tpr)
        logging.info("evaluation result: AUC=%4f" % (val_auc))
        writer.add_scalar("val/auc", val_auc, global_step=epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.module.state_dict(), '{}/ckp_model/model_best.pth'.format(exp_save_path))
        writer.add_scalar("val/best_model_result_auc", best_auc, global_step=epoch)
        return best_auc
