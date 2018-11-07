import torch
import numpy as np

def anchors_of_feature_map(stride, size, feature_map_shape):
    """
    Introduction
    ------------
        根据feature_map大小设置anchor
    Parameters
    ----------
        stride: anchor步长大小
        size: anchor大小
        feature_map_shape: feature_map大小
    Returns
    -------
        anchors: 生成的anchor
    """
    anchors = []
    height, width = feature_map_shape
    for row in range(height):
        center_y = row * stride + size // 2
        for col in range(width):
            center_x = col * stride + size // 2
            anchors.append( (center_x, center_y, size, size) )
    return anchors


def generate_anchors(anchor_stride, anchor_size, image_size):
    """
    Introduction
    ------------
        生成多尺度anchor box [center_x, center_y, width, height]
    Parameters
    ----------
        anchor_stride:
        anchor_size:
        image_size:
    Returns
    -------
        all_anchors: 所有尺度的anchor
    """
    all_anchors = []
    for index in range(len(anchor_stride)):
        anchors = []
        stride = anchor_stride[index]
        size = anchor_size[index]
        for row in range(image_size[0] // stride):
            center_y = row * stride + size // 2
            for col in range(image_size[1] // stride):
                center_x = col * stride + size // 2
                width = size
                height = size
                anchors.append([center_x / image_size[0], center_y / image_size[0], width / image_size[0], height / image_size[0]])
        all_anchors.append(anchors)
    return all_anchors


def compute_iou(anchors, gt_boxes):
    """
    Introduction
    ------------
        计算anchors和gt_boxes的iou
    Parameters
    ----------
        anchors: 生成的anchors
        gt_boxes: 真实坐标
    Returns
    -------
        iou: 每个anchor和每个gt_box的iou值
    """
    len_anchors = anchors.shape[0]
    len_gt_boxes = gt_boxes.shape[0]
    anchors = np.repeat(anchors, len_gt_boxes, axis=0)
    gt_boxes = np.vstack([gt_boxes] * len_anchors)

    x1 = np.maximum(anchors[:, 0], gt_boxes[:, 0])
    y1 = np.maximum(anchors[:, 1], gt_boxes[:, 1])
    x2 = np.minimum(anchors[:, 2], gt_boxes[:, 2])
    y2 = np.minimum(anchors[:, 3], gt_boxes[:, 3])

    y_zeros = np.zeros_like(y2.shape)
    x_zeros = np.zeros_like(x2.shape)

    intersect = np.maximum((y2 - y1), y_zeros) * np.maximum((x2 - x1), x_zeros)

    unit = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1]) + (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]) - intersect

    return (intersect / unit).reshape(len_anchors, len_gt_boxes)


def match_anchors(anchors, gt_boxes, positive_threshold = 0.3, negative_threshold = 0.1, least_pos_num = 50):
    """
    Introduction
    ------------
        根据anchor和真实bbox的iou筛选候选框
    Parameters
    ----------
        anchors: 生成的anchors坐标
        gt_boxes: 真实box坐标
        positive_threshold: 正样本的阈值
        negative_threshold: 负样本的阈值
        least_pos_num: 最小匹配的正样本数
    Returns
    -------
    """
    # 将anchors的坐标转换为[xmin, ymin, xmax, ymax]
    anchors = np.concatenate((anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2), 1)
    iou = compute_iou(anchors, gt_boxes)
    max_iou = iou.max(axis = 1)
    positive_anchor_indices = np.where(max_iou > positive_threshold)[0]
    negative_anchor_indices = np.where(max_iou < negative_threshold)[0]
    positive_iou = iou[positive_anchor_indices]
    matched_gt_box_indices = positive_iou.argmax(axis = 1)
    # 如果和ground truth box 匹配的box很少的话，在iou大于0.2的anchors中补充
    if len(matched_gt_box_indices) < least_pos_num:
        allowed_positive_anchor_indices = np.where(max_iou > 0.2)[0]
        top_n_sorted_indices = np.argsort(max_iou)[::-1][:least_pos_num]
        positive_anchor_indices = np.intersect1d(allowed_positive_anchor_indices, top_n_sorted_indices)
        positive_iou = iou[positive_anchor_indices]
        matched_gt_box_indices = positive_iou.argmax(axis = 1)
    return positive_anchor_indices, matched_gt_box_indices, negative_anchor_indices

