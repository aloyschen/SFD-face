import cv2
import torch
import config
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from model import build_SFD
from utils import nms
import torchvision.transforms as transforms
from anchor import generate_anchors

class Detector(object):
    def __init__(self, model_path, image_size, conf_threshold, nms_threshold, top_k):
        """
        Introduction
        ------------
            检测人脸
        Parameters
        ----------
            model_path: 模型路径
            image_size: 图片大小
            conf_threshold: 人脸检测阈值
            nms_threshold: nms阈值
            top_k: 保留前多少个
        """
        ckpt = torch.load(model_path, map_location='cpu')
        self.model = build_SFD()
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.image_size = image_size
        self.top_k = top_k

    def forward(self, images):
        loc, conf = self.model(images)
        num = loc.shape[0]
        anchors = generate_anchors(config.anchor_stride, config.anchor_size, self.image_size)
        anchors = torch.Tensor(np.vstack(anchors))
        print(conf[0])
        scores = nn.Softmax(dim = -1)(conf)
        print(scores[0][0])
        output = torch.zeros(num, 2, self.top_k, 5)
        scores = scores.reshape([num, -1, 2]).transpose(2, 1)
        for index in range(num):
            loc_t = loc[index].clone()
            conf_scores = scores[index].clone()
            bounding_boxes = torch.cat((
                anchors[:, :2] + loc_t[:, :2] * anchors[:, 2:],
                anchors[:, 2:] * torch.exp(loc_t[:, 2:])), 1)
            bounding_boxes[:, :2] -= bounding_boxes[:, 2:] / 2
            bounding_boxes[:, 2:] += bounding_boxes[:, :2]
            class_mask = conf_scores[1].gt(self.conf_threshold)
            scores = conf_scores[1][class_mask]
            if scores.shape[0] == 0:
                continue
            boxes = bounding_boxes[class_mask].detach()
            ids, count = nms(boxes, scores, self.nms_threshold, self.top_k)
            output[index, 1, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        return output

def predict(image_file, model_path):
    start = time.time()
    image = cv2.imread(image_file)
    image_resize = cv2.resize(image, (640, 640))
    image_transform = transforms.ToTensor()
    image_tensor = image_transform(image_resize).unsqueeze(0)
    detector = Detector(model_path, config.image_size, config.conf_threshold, config.nms_threshold, config.top_k)
    output = detector.forward(image_tensor)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for detect_image in output:
        for detect_class in detect_image[1:]:
            for object in detect_class:
                score = object[0]
                if score > config.conf_threshold:
                    print(score)
                    box = object[1:]
                    cv2.rectangle(image, (box[0] * 640, box[1] * 640), (box[2] * 640, box[3] * 640), [0, 255, 0], 2)
        print('detect time: {}'.format(time.time() - start))
        plt.imshow(image)
        plt.show()

if __name__ == '__main__':
    predict('./test.jpg', './checkpoint.pth.tar')


