import torch
import numpy as np
from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve

def cal_false_alarm(gt, preds, threshold=0.5):
    preds = list(preds.cpu().detach().numpy())
    gt = list(gt.cpu().detach().numpy())

    preds = np.repeat(preds, 16)
    preds[preds < threshold] = 0
    preds[preds >= threshold] = 1
    tn, fp, fn, tp = confusion_matrix(gt, preds, labels=[0, 1]).ravel()

    far = fp / (fp + tn)

    return far

def pad_tensor(tensor, max_seqlen):
    batch_size, num_frames, feature_dim = tensor.size()
    padding_length = max_seqlen - num_frames
    padded_tensor = torch.cat([tensor, torch.zeros(batch_size, padding_length, feature_dim)], dim=1)
    return padded_tensor

def test_func(dataloader, model, gt, dataset, test_bs):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        abnormal_preds = torch.zeros(0).cuda()
        abnormal_labels = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(gt.copy()).cuda()
        ab_pred = torch.zeros(0).cuda()
        ab_pred = torch.zeros(0).cuda()

        tmp_pred = torch.zeros(0).cuda()
        for i, (v_input, clip_input, label) in enumerate(dataloader):
            # with autocast():
            v_input = v_input.float().cuda(non_blocking=True)
            # print(v_input.shape)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            clip_input = clip_input[:, :torch.max(seq_len), :]
            # clip_input = pad_tensor(clip_input, torch.max(seq_len))
            
            clip_input = clip_input.float().cuda(non_blocking=True)
            
            if isinstance(label[0], str):
                label = [1]

            if max(seq_len) < 800:
                logits, _ = model(v_input, clip_input, seq_len)
                
                logits = torch.mean(logits, 0)
                logits = logits.squeeze(dim=-1)
                pred = torch.cat((pred, logits))
                if sum(label) == len(label):
                    ab_pred = torch.cat((ab_pred, logits))
                
            else:
                for v_in, cl_in, seq in zip(v_input, clip_input, seq_len):
                    v_in = v_in.unsqueeze(0)
                    cl_in = cl_in.unsqueeze(0)
                    seq = torch.tensor([seq]).cuda()
                    logits, _ = model(v_in, cl_in, seq)
                    tmp_pred = torch.cat((tmp_pred, logits))

                tmp_pred = torch.mean(tmp_pred, 0)
                tmp_pred = tmp_pred.squeeze(dim=-1)
                pred = torch.cat((pred, tmp_pred))
                if sum(label) == len(label):
                    ab_pred = torch.cat((ab_pred, tmp_pred))
                tmp_pred = torch.zeros(0).cuda()

        pred = list(pred.cpu().detach().numpy())
        fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
        roc_auc = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(rec, pre)
        
        ab_pred = list(ab_pred.cpu().detach().numpy())
        fpr, tpr, _ = roc_curve(list(gt)[:len(ab_pred)*16], np.repeat(ab_pred, 16))
        ab_roc_auc = auc(fpr, tpr)
        
        fpr, tpr, _ = roc_curve(list(gt)[:len(ab_pred)*16], np.repeat(ab_pred, 16))
        ab_roc_auc = auc(fpr, tpr)

        if dataset == 'ucf-crime':
            return roc_auc, ab_roc_auc
        elif dataset == 'xd-violence':
            return pr_auc, roc_auc#n_far
        elif dataset == 'shanghaiTech':
            return roc_auc, ab_roc_auc
        else:
            raise RuntimeError('Invalid dataset.')
