import os
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import ssl
 
ssl._create_default_https_context = ssl._create_unverified_context

class Resnet50FPN(nn.Module):
    def __init__(self, use_moco_pretrained=False):
        super(Resnet50FPN, self).__init__()
        if use_moco_pretrained:
            resnet = torchvision.models.resnet50(pretrained=False)
            if os.path.exists('weights/moco_v2_800ep_pretrain.pth.tar'):
                model_param = torch.load('weights/moco_v2_800ep_pretrain.pth.tar')
            else:
                # https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar
                model_param = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar')
            model_param = {k.replace('module.encoder_q.',''): v for k, v in model_param['state_dict'].items()}
            resnet.load_state_dict(model_param, strict=False)
        else:
            resnet = torchvision.models.resnet50(pretrained=True)
        children = list(resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]
    def forward(self, im_data):
        feat = OrderedDict()
        feat_map1 = self.conv1(im_data)
        feat_map2 = self.conv2(feat_map1)
        feat_map3 = self.conv3(feat_map2)
        feat_map4 = self.conv4(feat_map3)
        feat['map1'] = feat_map1
        feat['map2'] = feat_map2
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        return feat


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def weights_xavier_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
            