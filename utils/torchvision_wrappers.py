import torch
import sys
import os
from torchvision.models import ResNet
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import model_urls, Bottleneck

import torchvision.models as torchvision_models

__all__ = ['ResNetWithFeats', 'resnet50']

class ResNetWithFeats(ResNet):
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feats = x
        feats_src = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # apply FC for each feat
        feats_as_batch = feats.permute((0, 2, 3, 1)).contiguous().view((-1, feats.shape[1]))
        feats_as_batch = self.fc(feats_as_batch)
        feats_as_batch = feats_as_batch.view((feats.shape[0], feats.shape[2], feats.shape[3], feats_as_batch.shape[1]))
        feats_as_batch = feats_as_batch.permute((0, 3, 1, 2))
        feats = feats_as_batch

        return x, feats, feats_src

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetWithFeats(block, layers, **kwargs)
    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            state_dict = torch.load(pretrained)['state_dict']
        else:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        state_dict = {k[len('module.'):] if k.startswith('module.') else k : v for k, v in state_dict.items()}
        res = model.load_state_dict(state_dict, strict=False)
        print(res)
    return model

def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

class SplitBackbones(torch.nn.Module):
    def __init__(self, arch='resnet50', num_models=2, use_feats=False, **kwargs):
        super(SplitBackbones, self).__init__()
        self.models = []
        for iM in range(num_models):
            if use_feats:
                m__dict__ = sys.modules[__name__].__dict__
            else:
                m__dict__ = torchvision_models.__dict__
            self.models.append(m__dict__[arch](**kwargs))
        self.models = torch.nn.ModuleList(self.models)

    def forward(self, x, selector):
        res = []
        selector = selector.squeeze()
        sOut = None
        for iM in range(len(self.models)):
            if torch.sum(selector == iM).item() > 0:
                cur = self.models[iM](x[selector == iM])
                if not (isinstance(cur, tuple) or isinstance(cur, list)):
                    cur = (cur,)
                res.append(cur)
                if sOut is None:
                    sOut = [c.shape for c in cur]
            else:
                res.append([])
        ret = [torch.zeros(*([x.shape[0]] + list(s[1:]))).to(x.device) for s in sOut]
        for iM in range(len(res)):
            if len(res[iM]) > 0:
                for iR in range(len(ret)):
                    ret[iR][selector == iM] = res[iM][iR]
        return ret