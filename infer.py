import time
from test import *

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from utils import fixed_smooth, slide_smooth


def pad_tensor(tensor, max_seqlen):
    batch_size, num_frames, feature_dim = tensor.size()
    padding_length = max_seqlen - num_frames
    padded_tensor = torch.cat([tensor, torch.zeros(batch_size, padding_length, feature_dim)], dim=1)
    return padded_tensor

def infer_func(model, dataloader, gt, logger, cfg):
    st = time.time()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        abnormal_labels = torch.zeros(0).cuda()
        abnormal_preds = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(gt.copy()).cuda()
        
        tmp_pred = torch.zeros(0).cuda()
        
        seq_len_list = []

        for i, (v_input, clip_input, _) in enumerate(dataloader):
            v_input = v_input.float().cuda(non_blocking=True)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            clip_input = clip_input[:, :torch.max(seq_len), :]
            clip_input = pad_tensor(clip_input, torch.max(seq_len))
            
            clip_input = clip_input.float().cuda(non_blocking=True)
            
            seq_len_list.append(seq_len[0])
            
            if max(seq_len) < 1200:
                logits, _ = model(v_input, clip_input, seq_len)
                tmp_pred = torch.cat((tmp_pred, logits))
            else:
                for v_in, cl_in, seq in zip(v_input, clip_input, seq_len):
                    v_in = v_in.unsqueeze(0)
                    cl_in = cl_in.unsqueeze(0)
                    seq = torch.tensor([seq]).cuda()
                    logits, _ = model(v_in, cl_in, seq)
                    tmp_pred = torch.cat((tmp_pred, logits))
            
            assert tmp_pred.shape[0] == cfg.test_bs
            logits = tmp_pred
            logits = torch.mean(logits, 0)
            logits = logits.squeeze(dim=-1)
            
            seq = len(logits)
            if cfg.smooth == 'fixed':
                logits = fixed_smooth(logits, cfg.kappa)
            elif cfg.smooth == 'slide':
                logits = slide_smooth(logits, cfg.kappa)
            else:
                pass
            logits = logits[:seq]

            pred = torch.cat((pred, logits))
            labels = gt_tmp[: seq_len[0]*16]
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
            else:
                abnormal_labels = torch.cat((abnormal_labels, labels))
                abnormal_preds = torch.cat((abnormal_preds, logits))
            gt_tmp = gt_tmp[seq_len[0]*16:]
        
            tmp_pred = torch.zeros(0).cuda()

        pred = list(pred.cpu().detach().numpy())
        abnormal_preds = list(abnormal_preds.cpu().detach().numpy())
        
        far = cal_false_alarm(normal_labels, normal_preds)
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                                     confusion_matrix)

        # gt, pred
        df = pd.DataFrame({'gt': gt, 'pred': np.repeat(pred, 16)})
        df.to_csv(f'./csv/{cfg.dataset}_gt_pred.csv', index=False)
        # gtが0の時のpredの値
        pred_gt_0 = [p for p, g in zip(np.repeat(pred, 16), gt) if g == 0]

        # gtが1の時のpredの値
        pred_gt_1 = [p for p, g in zip(np.repeat(pred, 16), gt) if g == 1]

        # ヒストグラムの作成
        plt.figure()
        plt.hist(pred_gt_0, bins=50, alpha=0.3, log=True, label='gt = 0', color='blue')
        plt.hist(pred_gt_1, bins=50, alpha=0.3, log=True, label='gt = 1', color='red')

        # グラフのラベル付け
        plt.xlabel('pred (0 to 1)')
        plt.ylabel('Frequency')
        plt.title('Histogram of pred values')
        plt.legend()

        # 画像を保存
        plt.savefig(f'./csv/{cfg.dataset}_histogram.png')
        plt.close()

        plt.figure()
        # Convert predictions to binary using threshold of 0.5
        binary_pred = [1 if p >= 0.5 else 0 for p in np.repeat(pred, 16)]

        # Calculate confusion matrix
        cm = confusion_matrix(gt, binary_pred)
        acc = accuracy_score(gt, binary_pred)

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)

        # Save confusion matrix plot
        plt.title('Confusion Matrix at 0.5 Threshold')
        plt.savefig(f'./csv/{cfg.dataset}_confusion_matrix.png')
        plt.close()

        # all
        fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f'./csv/{cfg.dataset}_roc_curve.png')  # ROC曲線の画像を保存
        plt.close()

        
        # anomaly
        fpr, tpr, _ = roc_curve(list(gt)[:len(abnormal_preds)*16], np.repeat(abnormal_preds, 16))
        ab_roc_auc = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(rec, pre)
        plt.figure()
        plt.plot(rec, pre, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(f'./csv/{cfg.dataset}_pr_curve.png')  # PR曲線の画像を保存
        plt.close()

        # acc log
        logger.info('offline Accuracy:{:.4f}'.format(acc))

    time_elapsed = time.time() - st
    logger.info('offline AUC:{:.4f} Anomaly-AUC:{:.4f} AP:{:.4f} FAR:{:.4f} | Complete in {:.0f}m {:.0f}s\n'.format(
        roc_auc, ab_roc_auc, pr_auc, far, time_elapsed // 60, time_elapsed % 60))
