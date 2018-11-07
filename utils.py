import torch
import numpy as np

def my_collate_fn(batch):
    """
    Introduction
    ------------
        对dataset解析
    parameters
    ----------
        batch: 每个batch的数据
    """
    images = torch.stack(list(map(lambda x: torch.Tensor(x[0]), batch)))
    coordinates = list(map(lambda x: x[1], batch))
    return images, coordinates



def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors, 4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def encode(matches, priors, variances):
    """
    Introduction
    ------------
        对正样本的prior box坐标进行encode变换，使用每个prior box和truth box的偏移做为训练数据集的坐标
    Parameters
    ----------
        matches: 每个prior box匹配的truth box坐标
        priors: 预选框坐标
        variances: 个人理解是为了控制box坐标范围
    Returns
    -------
        encoded boxes: 转换后的坐标
    """
    box_xy = (matches[:, :2] + matches[:, 2:]) / 2 - priors[:, :2]
    box_xy /= (variances[0] * priors[:, 2:])
    box_wh = (matches[:, 2:] - matches[:, :2]) / priors[:, 2:]
    box_wh = torch.log(box_wh) / variances[1]
    return torch.cat([box_xy, box_wh], 1)

def decode(loc, priors, variances):
    """
    Introduction
    ------------
        对模型预测出的坐标转换回来，对应encode的逆变换
    Parameters
    ----------
        loc: 预测box坐标值
        priors: 预选框坐标值
        variances: 坐标修正系数
    Returns
    -------
        boxes: 变换后的坐标
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes