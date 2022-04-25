from PIL import ImageFilter, Image
import random
import torch
from skimage import feature
from skimage.morphology import dilation, disk


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, args):
        self.base_transform = base_transform
        self.args = args

    @staticmethod
    def trans(t, q, k):
        is_edge_q = torch.zeros((1,), dtype=torch.int)
        is_edge_k = torch.ones((1,), dtype=torch.int)
        q = t[0](q)
        k = t[1](k)
        return q.unsqueeze(0), k.unsqueeze(0), is_edge_q.unsqueeze(0), is_edge_k.unsqueeze(0)

    def __call__(self, x):
        q, k, is_edge_q, is_edge_k = self.trans(self.base_transform, x, x)
        q2, k2, is_edge_q2, is_edge_k2 = self.trans(self.base_transform[::-1], x, x)
        is_edge_q2[:] = 1
        is_edge_k2[:] = 0
        q = torch.cat([q, q2], dim=0)
        k = torch.cat([k, k2], dim=0)
        is_edge_q = torch.cat([is_edge_q, is_edge_q2], dim=0)
        is_edge_k = torch.cat([is_edge_k, is_edge_k2], dim=0)

        return [q, k, is_edge_q, is_edge_k]


class TwoCropsTransformPlusCanny:
    """Take two random crops of one image as the query and key and apply Canny on the queries"""

    def __init__(self, base_transform, args):
        self.base_transform = base_transform
        self.args = args

    @staticmethod
    def trans(t, q, k, isedge):

        is_edge_q = torch.zeros((1,), dtype=torch.int)
        ixTk = torch.ones((1,), dtype=torch.int)
        q = t[0](q)
        k = t[1](k)
        if isedge:
            assert (isinstance(t, tuple) and len(t) == 3)
            c = t[2](q)
        else:
            c = q
        return q.unsqueeze(0), k.unsqueeze(0), is_edge_q.unsqueeze(0), ixTk.unsqueeze(0), c.unsqueeze(0)

    def __call__(self, x):
        q, k, ixTq, ixTk, _ = self.trans(self.base_transform, x, x, False)
        q2, k2, ixTq2, ixTk2, c = self.trans((self.base_transform[1], self.base_transform[0], self.base_transform[2]),
                                             x, x, True)
        ixTq2[:] = 1
        ixTk2[:] = 0
        q = torch.cat([q, q2], dim=0)
        k = torch.cat([k, k2], dim=0)
        ixTq = torch.cat([ixTq, ixTq2], dim=0)
        ixTk = torch.cat([ixTk, ixTk2], dim=0)

        return [q, k, ixTq, ixTk, c]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(.1, 2.)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class StretchValues(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(img):
        img -= torch.min(img)
        img /= (torch.max(img)+1e-14)
        return img


class ToEdges2D(torch.nn.Module):
    def __init__(self, sigma='1.0', dil='0'):
        super().__init__()
        self.sigma = [float(x) for x in sigma.split(',')]
        self.dil = [int(x) for x in dil.split(',')]

    def forward(self, img):
        img = torch.mean(img, dim=0).numpy()
        sig = self.sigma[torch.randint(len(self.sigma), (1,))[0]]
        edg = feature.canny(img, sigma=sig)
        dl = self.dil[torch.randint(len(self.dil), (1,))[0]]
        if dl > 0:
            edg = dilation(edg, disk(dl))
        return torch.Tensor(edg).unsqueeze(dim=0)


class ToEdges(torch.nn.Module):
    def __init__(self, sigma='1.0', dil='0'):
        super().__init__()
        self.sigma = [float(x) for x in sigma.split(',')]
        self.dil = [int(x) for x in dil.split(',')]

    def forward(self, img):
        img = torch.mean(img, dim=0).numpy()
        sig = self.sigma[torch.randint(len(self.sigma), (1,))[0]]
        edg = feature.canny(img, sigma=sig)
        dl = self.dil[torch.randint(len(self.dil), (1,))[0]]
        if dl > 0:
            edg = dilation(edg, disk(dl))
        img_out = torch.Tensor(edg).unsqueeze(dim=0).repeat([3, 1, 1])
        return img_out

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)