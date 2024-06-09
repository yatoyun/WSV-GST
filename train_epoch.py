import torch
from utils import *
from loss.loss import *

def interpolate_frames(x, seq_len):
    bs, max_len, _ = x.size()

    for idx in range(bs):
        valid_length = seq_len[idx]

        # padding部分のフレーム数を計算
        padding_frames_count = max_len - valid_length

        # 補間フレームを生成
        if padding_frames_count > 0:
            last_valid_frame = x[idx, valid_length - 1]
            second_last_valid_frame = x[idx, valid_length - 2]
            
            alpha = torch.linspace(0, 1, padding_frames_count + 2, device=x.device)[1:-1].unsqueeze(1)
            
            interpolated = alpha * last_valid_frame + (1 - alpha) * second_last_valid_frame
            
            # padding部分を補間フレームで置き換え
            x[idx, valid_length:] = interpolated

    return x


def to_cuda(v_input, clip_input, t_input, label, multi_label):
    v_input = v_input.float().cuda(non_blocking=True)
    clip_input = clip_input.float().cuda(non_blocking=True)
    t_input = t_input.float().cuda(non_blocking=True)
    label = label.float().cuda(non_blocking=True)
    multi_label = multi_label.cuda(non_blocking=True)
    
    return v_input, clip_input, t_input, label, multi_label
    
def train_func(normal_iter, anomaly_iter, model, optimizer, criterion, criterion2, criterion3, logger_wandb, lamda=0, alpha=0, margin=100.0):
    v_ninput, clip_ninput, t_ninput, nlabel, multi_nlabel = normal_iter #next(normal_iter)
    v_ainput, clip_ainput, t_ainput, alabel, multi_alabel = anomaly_iter #next(anomaly_iter)
    with torch.set_grad_enabled(True):
        model.train()
        
        v_ninput, clip_ninput, t_ninput, nlabel, multi_nlabel = to_cuda(v_ninput, clip_ninput, t_ninput, nlabel, multi_nlabel)
        v_ainput, clip_ainput, t_ainput, alabel, multi_alabel = to_cuda(v_ainput, clip_ainput, t_ainput, alabel, multi_alabel)
        # cat
        v_input = torch.cat((v_ninput, v_ainput), 0)
        t_input = torch.cat((t_ninput, t_ainput), 0)
        clip_input = torch.cat((clip_ninput, clip_ainput), 0)
        label = torch.cat((nlabel, alabel), 0)
        multi_label = torch.cat((multi_nlabel, multi_alabel), 0)
        # seq_len
        seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
        v_input = v_input[:, :torch.max(seq_len), :]
        clip_input = clip_input[:, :torch.max(seq_len), :]
        
        
        v_input = interpolate_frames(v_input, seq_len)
        clip_input = interpolate_frames(clip_input, seq_len)

        logits, x_k, output_MSNSD = model(v_input, clip_input, seq_len)
        
        v_feat = x_k["x"]
        x_k["frame"] = logits
        
        # Prompt-Enhanced Learning
        logit_scale = model.logit_scale.exp()
        video_feat, token_feat, video_labels = get_cas(v_feat, t_input, logits, multi_label)
        v2t_logits, v2v_logits = create_logits(video_feat, token_feat, logit_scale)
        
        ground_truth = gen_label(video_labels)
        loss2 = KLV_loss(v2t_logits, ground_truth, criterion2)

        loss1 = CLAS2(logits, label, seq_len, criterion)
        
        UR_loss = criterion3(x_k, label, seq_len)[0]
        # mgc loss
        loss_criterion = mgc_loss(margin)
        mg_loss = loss_criterion(output_MSNSD)
        
        # loss 1
        loss1 = loss1 + mg_loss

        loss = loss1 + lamda * loss2 + alpha * UR_loss
        
        logger_wandb.log({"loss": loss.item(), "loss1":loss1.item(), "loss2": loss2.item(), "loss3": UR_loss.item()})


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss1.item(), loss2.item(), UR_loss.item()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=100.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class mgc_loss(torch.nn.Module):
    def __init__(self, margin):
        super(mgc_loss, self).__init__()
        self.criterion = torch.nn.BCELoss()
        self.contrastive = ContrastiveLoss(margin)



    def forward(self, output):
        nor_feamagnitude = output["nor_feamagnitude"]
        abn_feamagnitude = output["abn_feamagnitude"]
        loss_con = self.contrastive(torch.norm(abn_feamagnitude, p=1, dim=2), torch.norm(nor_feamagnitude, p=1, dim=2),
                                    1)  # try tp separate normal and abnormal
        loss_total = 0.001 * (0.01 * loss_con)
        
        return loss_total

