import torch
import torch.nn as nn
import torch.nn.functional as F
from anchor import match_anchors
from utils import encode

class MultiBoxLoss(nn.Module):
    def __init__(self):
        super(MultiBoxLoss, self).__init__()
        self.positive_threshold = 0.35
        self.negative_threshold = 0.1
        self.least_pos_num = 100
        self.neg_pos_anchor_num_ratio = 3

    def forward(self, predictions, anchors, targets):
        loc, conf = predictions
        num = loc.shape[0]
        total_t = []
        total_gt = []
        total_effective_pred = []
        total_target = []
        for index in range(num):
            gt_boxes = torch.Tensor(targets[index]).float().cuda()
            conf_t = conf[index]
            loc_t = loc[index]
            if len(gt_boxes) == 0:
                continue
            pos_indices, gt_bboxes_indices, neg_indices = match_anchors(anchors, gt_boxes, positive_threshold = self.positive_threshold, negative_threshold = self.negative_threshold)
            if len(pos_indices) == 0:
                continue
            neg_num = len(pos_indices) * self.neg_pos_anchor_num_ratio
            neg_cls_preds = conf_t[neg_indices]
            neg_indices = torch.sort(neg_cls_preds[:, 0])[1][:neg_num]
            pos_anchors = anchors[pos_indices]
            total_t.append(loc_t[pos_indices])
            matched_bboxes = gt_boxes[gt_bboxes_indices]
            box_xy = (matched_bboxes[:, :2] + matched_bboxes[:, 2:]) / 2 - pos_anchors[:, :2]
            box_xy /= pos_anchors[:, 2:]
            box_wh = torch.log((matched_bboxes[:, 2:] - matched_bboxes[:, :2]) / pos_anchors[:, 2:])
            gt = torch.cat([box_xy, box_wh], 1)
            total_gt.append(gt)
            pos_targets = torch.ones(len(pos_indices)).long().cuda()
            neg_targets = torch.zeros(len(neg_indices)).long().cuda()
            effective_preds = torch.cat((conf_t[pos_indices], neg_cls_preds[neg_indices]))
            all_boxes = torch.cat((pos_targets, neg_targets))
            shuffle_indexes = torch.randperm(effective_preds.shape[0])
            effective_preds = effective_preds[shuffle_indexes]
            all_boxes = all_boxes[shuffle_indexes]
            total_effective_pred.append(effective_preds)
            total_target.append(all_boxes)
        if len(total_t) == 0:
            return None, None
        total_t = torch.cat(total_t)
        total_gt = torch.cat(total_gt)
        total_targets = torch.cat(total_target)
        total_effective_pred = torch.cat(total_effective_pred)
        loss_conf = F.cross_entropy(total_effective_pred, total_targets)
        loss_loc = F.smooth_l1_loss(total_t, total_gt)

        return loss_loc, loss_conf


# def box_iou(box1, box2):
#     """
#     Introduction
#     ------------
#         计算两个box的iou
#     Parameters
#     ----------
#         box1: box的坐标[xmin, ymin, xmax, ymax], shape[box1_num, 4]
#         box2: box的坐标[xmin, ymin, xmax, ymax], shape[box2_num, 4]
#     Returns
#     -------
#         iou: 两个box的iou数值, shape[box1_num, box2_num]
#     """
#     box_num1 = box1.shape[0]
#     box_num2 = box2.shape[0]
#     max_xy = torch.min(box1[:, 2:].unsqueeze(1).expand(box_num1, box_num2, 2), box2[:, 2:].unsqueeze(0).expand(box_num1, box_num2, 2))
#     min_xy = torch.max(box1[:, :2].unsqueeze(1).expand(box_num1, box_num2, 2), box2[:, :2].unsqueeze(0).expand(box_num1,box_num2, 2))
#     inter_xy = torch.clamp((max_xy - min_xy), min = 0)
#     inter_area = inter_xy[:, :, 0] * inter_xy[:, :, 1]
#     box1_area = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])).unsqueeze(1).expand_as(inter_area)
#     box2_area = ((box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])).unsqueeze(0).expand_as(inter_area)
#     iou = inter_area / (box1_area + box2_area - inter_area)
#     return iou
#
#
# def match_priors(truths, priors, labels, threshold, variances, loc_t, conf_t, idx, least_pos_num):
#     """
#     Introduction
#     ------------
#         匹配priors box和truth box, 计算每个prior box和truth box重合iou，如果最大iou大于阈值则认为是正样本
#     Parameters
#     ----------
#         truths: ground truth box坐标
#         priors: 预选框坐标
#         labels: ground truth box对应的类别
#         threshold: iou阈值
#         variances: 坐标变换修正
#         loc_t: 正样本坐标
#         conf_t: 每个prior box的类别，0为不含物体的背景
#         idx: 每张训练图片的index
#         least_pos_num: 正样本最小数量
#     """
#     # 将priors坐标转换为[xmin, ymin, xmax, ymax]
#     priors_ = torch.cat((priors[:, :2] - priors[:, 2:] / 2, priors[:, :2] + priors[:, 2:] / 2), 1)
#     iou = box_iou(truths, priors_)
#     # 找到每个truth box对应的iou最大的prior box
#     best_priors_iou, best_priors_idx = iou.max(1, keepdim = True)
#     # 找到每个prior box对应的iou最大的truth box
#     best_truth_iou, best_truth_idx = iou.max(0, keepdim = True)
#     best_truth_iou.squeeze_(0)
#     best_truth_idx.squeeze_(0)
#     best_priors_idx.squeeze_(1)
#     best_priors_iou.squeeze_(1)
#     # 将和truth box重合度最高的prior box标记出来
#     best_truth_iou.index_fill(0, best_priors_idx, 2)
#     # 确保每个prior box只对应一个truth box
#     for j in range(best_priors_idx.shape[0]):
#         best_truth_idx[best_priors_idx[j]] = j
#     # 获取每个prior box对应的truth box
#     matches = truths[best_truth_idx]
#     matches[best_truth_iou == 0] = 0
#     conf = labels[best_truth_idx]
#     conf[best_truth_iou < threshold] = 0
#     loc = encode(matches, priors, variances)
#     loc_t[idx] = loc
#     conf_t[idx] = conf
#
#
# class MultiBoxLoss(nn.Module):
#     def __init__(self, num_classes, overlap_thresh, neg_pos_ratio, least_pos_num = 50):
#         """
#         Introduction
#         ------------
#             计算多尺度loss
#         Parameters
#         ----------
#             num_classes: 数据集类别数量
#         """
#         super(MultiBoxLoss, self).__init__()
#         self.num_classes = num_classes
#         self.threshold = overlap_thresh
#         self.variances = [0.1, 0.2]
#         self.neg_pos_ratio = neg_pos_ratio
#         self.least_pos_num = least_pos_num
#
#
#     def forward(self, predictions, priors, targets):
#         """
#         Parameters
#         ----------
#             predictions: 包含位置预测值和类别预测值
#             priors: 每个feature map上生成的prior box
#             targets: 每张图片上对应的box坐标和类别id，最后一维是类别坐标
#         """
#         loc_data, conf_data = predictions
#         num = loc_data.shape[0]
#         num_priors = priors.shape[0]
#         priors = torch.Tensor(priors)
#         loc_t = torch.Tensor(num, num_priors, 4)
#         conf_t = torch.LongTensor(num, num_priors).cuda()
#         for idx in range(num):
#             truths = targets[idx][:, :-1]
#             labels = targets[idx][:, -1]
#             truths = torch.Tensor(truths)
#             labels = torch.Tensor(labels)
#             defaults = priors
#             match_priors(truths, defaults, labels, self.threshold, self.variances, loc_t, conf_t, idx, self.least_pos_num)
#         pos = conf_t > 0
#         # 广播扩充维度
#         pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
#         loc_pred = loc_data[pos_idx].view(-1, 4)
#         loc_truth = loc_t[pos_idx].view(-1, 4)
#         # 计算正样本的location位置损失
#         loss_location = F.smooth_l1_loss(loc_pred, loc_truth, reduction = 'sum')
#         # 计算所有样本的类别损失
#         batch_conf = conf_data.reshape([-1, self.num_classes])
#         loss_conf = F.cross_entropy(batch_conf, conf_t.reshape([-1]), ignore_index = -1, reduction = 'none')
#         loss_conf = loss_conf.reshape([num, -1])
#         # 将负样本对应的loss_conf排序
#         pos_loss_conf = loss_conf[pos]
#         loss_conf[pos] = 0
#         _, loss_idx = loss_conf.sort(1, descending = True)
#         _, idx_rank = loss_idx.sort(1)
#         num_pos = pos.long().sum(1, keepdim = True)
#         num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max = pos.shape[1] - 1)
#         neg = idx_rank < num_neg.expand_as(idx_rank)
#         neg_loss_conf = loss_conf[neg]
#         loss_conf = pos_loss_conf.sum() + neg_loss_conf.sum()
#         N = num_pos.data.sum().float()
#         if N == 0:
#             return None, None
#         loss_conf = loss_conf / N
#         loss_location = loss_location / N
#         return loss_location, loss_conf
