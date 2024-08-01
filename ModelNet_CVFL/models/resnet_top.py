import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['RESNET_top', 'resnet_top']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class RESNET_top(nn.Module):

    def __init__(self, num_classes=1000, num_clients=4):
        super(RESNET_top, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(16 * num_clients, num_classes),
        )
        #self.classifier = nn.Sequential(
        #    nn.Linear(num_classes * num_clients, 100),
        #    nn.Linear(100, 100),
        #    nn.Linear(100, num_classes),
        #)

    def forward(self, x):
        pooled_view = self.classifier(x)
        return pooled_view

from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module

class Server_L2(Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        embedding_size: int = 128
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


def resnet_top(pretrained=False, **kwargs):
    r"""MVCNN model architecture from the
    `"Multi-view Convolutional..." <hhttp://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = RESNET_top(**kwargs)
    # if pretrained:
    #     pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
    #     model_dict = model.state_dict()
    #     # 1. filter out unnecessary keys
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    #     # 2. overwrite entries in the existing state dict
    #     model_dict.update(pretrained_dict)
    #     # 3. load the new state dict
    #     model.load_state_dict(model_dict)
    # return model
    return Server_L2()
