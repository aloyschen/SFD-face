import config
import torch
import torch.nn as nn
import torch.nn.init as init

def build_vgg16(cfg, batch_norm = False):
    """
    Introduction
    ------------
        构建vgg16模型结构
    Parameters
    ----------
        cfg: 模型层结构参数
        batch_norm: 是否使用batch norm
    Returns
    -------
        vgg16模型结构
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = v
    return layers

def multiScale_extra():
    """
    Introduction
    ------------
        多尺度特征提取
    """
    net = []
    net.append(nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1))
    net.append(nn.Conv2d(1024, 1024, kernel_size = 1, stride = 1))
    net.append(nn.Conv2d(1024, 256, kernel_size = 1, stride = 1))
    net.append(nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = 1))
    net.append(nn.Conv2d(512, 128, kernel_size = 1, stride = 1))
    net.append(nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1))
    return net


def multi_loc_conf(vgg, extras):
    """
    Introduction
    ------------
        通过对vgg和额外的特征层卷积获取box的坐标和概率
    Parameters
    ----------
        vgg: 基础vgg模型
        extras: 额外特征提取层
    """
    loc_layers = []
    conf_layers = []
    vgg_source = [14, 21, 28]
    for index, value in enumerate(vgg_source):
        loc_layers.append(nn.Conv2d(vgg[value].out_channels, 4, kernel_size = 3, padding = 1))
        conf_layers.append(nn.Conv2d(vgg[value].out_channels, 2, kernel_size = 3, padding = 1))
    for index, value in enumerate(extras):
        if index % 2 == 1:
            loc_layers.append(nn.Conv2d(value.out_channels, 4, kernel_size = 3, padding = 1))
            conf_layers.append(nn.Conv2d(value.out_channels, 2, kernel_size = 3, padding = 1))
    return loc_layers, conf_layers


class Scale(nn.Module):

    def __init__(self, input_channels, initialized_factor):
        """
        Introduction
        ------------
        Parameters
        ----------
            input_channels: 输入通道数
            initialized_factor: 缩放系数
        """
        super().__init__()
        self.input_channels = input_channels
        self.eps = 1e-10
        self.factor = nn.Parameter(torch.Tensor(self.input_channels))
        nn.init.constant_(self.factor, initialized_factor)

    def forward(self, x):
        norm = torch.pow(x, 2).sum(dim = 1, keepdim = True).sqrt() + self.eps
        return x / norm * self.factor.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)


class SFD(nn.Module):
    def __init__(self, base_model, extras, head):
        """
        Introduction
        ------------
            构建人脸检测SFD模型
        """
        super(SFD, self).__init__()
        self.base_model = nn.ModuleList(base_model)
        self.extra_model = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.norm3_3 = Scale(256, 10)
        self.norm4_3 = Scale(512, 8)
        self.norm5_3 = Scale(512, 5)
        for m in self.modules():
            self.weights_init(m)


    def forward(self, x):
        sources = []
        loc = []
        conf = []
        # 添加vgg模型中三层特征160x160x256 80x80x512 40x40x512
        for index in range(16):
            x = self.base_model[index](x)
        sources.append(self.norm3_3(x))
        for index in range(16, 23):
            x = self.base_model[index](x)
        sources.append(self.norm4_3(x))
        for index in range(23, 30):
            x = self.base_model[index](x)
        sources.append(self.norm5_3(x))
        # 额外特征提取结构
        x = self.base_model[30](x)
        for index, layer in enumerate(self.extra_model):
            x = nn.ReLU()(layer(x))
            if index % 2 == 1:
                sources.append(x)
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1))
            conf.append(c(x).permute(0, 2, 3, 1))
        loc = torch.cat([layer.reshape(layer.shape[0], -1) for layer in loc], 1)
        conf = torch.cat([layer.reshape(layer.shape[0], -1) for layer in conf], 1)
        output = (
            loc.reshape(loc.shape[0], -1, 4),
            conf.reshape(conf.shape[0], -1, 2)
        )
        return output

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)

def build_SFD():
    """
    Introduction
    ------------
        构建SFD模型
    """
    vgg = build_vgg16(config.cfg['vgg16'])
    extras = multiScale_extra()
    model = SFD(vgg, extras, multi_loc_conf(vgg, extras))
    return model

