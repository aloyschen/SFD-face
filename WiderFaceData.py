import cv2
import os
import random
import numpy as np
import torch.utils.data as data

class WiderFaceDataset(data.Dataset):
    def __init__(self, root, annotation_file, training, transform):
        """
        Introduction
        ------------
            构建wider face人脸检测数据集
        Parameters
        ----------
            root: 图片数据集路径
            annotation_file: 数据集标注文件
            training: 是否为训练的标志位
            transform: 数据集变换
        """
        self.root = root
        self.annotation_file = annotation_file
        self.training = training
        self.image_data = []
        self.targets = []
        self.image_size = 640
        self.transform = transform

        with open(annotation_file) as f:
            lines = f.readlines()
            cursor = 0

            while True:
                if len(lines) == cursor:
                    break
                image_file = lines[cursor].strip()
                self.image_data.append(image_file)
                face_count = int(lines[cursor + 1])
                bboxes_data = lines[cursor + 2:cursor + face_count + 2]
                coordinates = []
                for bbox_data in bboxes_data:
                    bbox = bbox_data.split()
                    coordinates.append((
                        int(bbox[0]), int(bbox[1]),
                        int(bbox[0]) + int(bbox[2]),
                        int(bbox[1]) + int(bbox[3])))
                self.targets.append(coordinates)
                cursor = cursor + 2 + face_count
        self.num_samples = len(self.image_data)


    def __getitem__(self, index):
        """
        Introduction
        ------------
            获取指定index的数据样本
        Parameters
        ----------
            index: 数据集中样本的index
        Returns
        -------
        """
        image_file = self.image_data[index]
        image = cv2.imread(os.path.join(self.root, image_file))
        targets = self.targets[index].copy()
        if self.training:
            image, targets = self.random_crop(image, targets)
            image = self.random_bright(image)
            image, targets = self.random_flip(image, targets)
        h, w, _ = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        targets = np.array(list(map(lambda x: [x[0] / w, x[1] / h, x[2] / w, x[3] / h], targets)))
        for transform in self.transform:
            image = transform(image)
        return image, targets


    def random_getim(self):
        """
        Introduction
        ------------
            随机获取数据集中的一张图片和对应的人脸坐标
        Returns
        -------
            image: 数据集图片
            boxes: 图片对应的
        """
        idx = random.randrange(0, self.num_samples)
        fname = self.image_data[idx]
        img = cv2.imread(os.path.join(self.root + fname))
        target = self.targets[idx].copy()
        return img, target

    def __len__(self):
        return self.num_samples


    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            new_boxes = []
            for box in boxes:
                new_boxes.append([im_lr.shape[1] - box[0], box[1], im_lr.shape[1] - box[2], box[3]])
            return im_lr, new_boxes
        return im, boxes


    def random_crop(self, image, targets, ratio = 1., keep_area_threshold = 0.5):
        """
        Introduction
        ------------
            对数据集图片随机裁剪层正方形, 同时舍弃裁剪后面积较小的box
        Parameters
        ----------
            image: 需要裁剪的图片
            targets: 图片对应的标注
            ratio: 裁剪正方形边长系数
            keep_area_threshold:
        """
        size = image.shape[:2]
        short_size = np.min(size)
        square_size = int(short_size * ratio)

        n_top = int((image.shape[0] - square_size) * random.random())
        n_left = int((image.shape[1] - square_size) * random.random())
        n_bottom = n_top + square_size
        n_right = n_left + square_size

        cropped_image = image[n_top:n_bottom, n_left:n_right]

        new_targets = []
        for target in targets:
            height = target[3] - target[1]
            width = target[2] - target[0]
            n_width = max(min(target[2], n_right) - max(target[0], n_left), 0)
            n_height = max(min(target[3], n_bottom) - max(target[1], n_top), 0)

            # wider face中有些人脸框标注为0
            if (width * height) == 0:
                continue
            area_in_crop = (n_width * n_height) / (width * height)

            if area_in_crop < keep_area_threshold:
                continue

            new_targets.append([
                max(target[0] - n_left, 0),
                max(target[1] - n_top, 0),
                max(target[2] - n_left, 0),
                max(target[3] - n_top, 0)
            ])
        if len(new_targets) == 0:
            return image, targets
        return cropped_image, new_targets

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im


