import os
import config
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from Modules import MultiBoxLoss
from anchor import generate_anchors
from utils import my_collate_fn
import torch.optim as optim
from model import build_SFD
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from WiderFaceData import WiderFaceDataset

# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index


def train():
    """
    Introduction
    ------------
        训练sfd模型
    """
    model = build_SFD()
    vgg_weights = torch.load(config.vgg_path)
    model.base_model.load_state_dict(vgg_weights)
    model.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True
    optimizer = optim.SGD(model.parameters(), lr = config.learning_rate, weight_decay = config.weight_decay)
    train_dataset = WiderFaceDataset(config.wider_face_train_dir, config.wider_face_train_annotations, training = True, transform = [transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size = config.train_batch, shuffle = True, num_workers = 0, pin_memory = True, collate_fn = my_collate_fn)
    result = generate_anchors(config.anchor_stride, config.anchor_size, config.image_size)
    anchors = torch.Tensor(np.vstack(list(map(lambda x: np.array(x), result)))).cuda()
    criterion = MultiBoxLoss()
    total_iter = len(train_loader)
    for epoch in range(config.Epoch_num):
        for index, (images, gt_bboxes) in enumerate(train_loader):
            images = images.cuda()
            predictions = model(images)
            loss_loc, loss_conf = criterion(predictions, anchors, gt_bboxes)
            if loss_loc is None:
                continue
            loss = loss_conf + loss_loc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if index % config.print_iter == 0:
                print("[epoch:{}][iter:{}][total:{}] loss_class {:.4f} - loss_reg {:.4f} - total {:.4f}".format(epoch, index, total_iter, loss_conf.data, loss_loc.data, loss.data))
        if epoch % 3 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, config.checkpoint_dir + 'checkpoint.pth.tar')

if __name__ == '__main__':
    train()