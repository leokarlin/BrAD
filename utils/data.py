import os
import numpy as np
import torch.utils.data

import utils.domainnet
import torchvision.datasets as datasets


class DataCoLoaderIterator(object):
    def __init__(self, data_loaders, args):
        self.args = args
        self.data_loaders = data_loaders
        self.data_loaders_iters = [iter(d) for d in data_loaders]
        self.nLoaders = len(self.data_loaders_iters)
        self.done = np.zeros((self.nLoaders,), dtype=bool)

    def __next__(self):
        ret = []
        for iit, it in enumerate(self.data_loaders_iters):
            try:
                x = next(it)
            except StopIteration:
                x = None
                self.done[iit] = True
            ret.append(x)
        if not np.all(self.done):
            for iR, x in enumerate(ret):
                if x is None:
                    # shuffle for DDP
                    try:
                        self.data_loaders[iR].sampler.set_epoch(np.random.randint(100000))
                    except:
                        pass
                    self.data_loaders_iters[iR] = iter(self.data_loaders[iR])
                    ret[iR] = next(self.data_loaders_iters[iR])
        else:
            raise StopIteration()

        return ret


class DataCoLoader(object):
    def __init__(self, data_loaders, args):
        self.args = args
        self.data_loaders = data_loaders

    def __iter__(self):
        return DataCoLoaderIterator(self.data_loaders, self.args)

    def __len__(self):
        return np.max([len(x) for x in self.data_loaders])


# wrapper to a given Dataset also returning the absolute index of the elements as well as the path
class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, data, *args, **kwargs):
        self.data = data
        super(IndexedDataset, self).__init__()

    def __getitem__(self, index_):
        index = self.get_index(index_)

        img, target = self.data[index]
        if isinstance(self.data, utils.domainnet.Dataset):
            path = os.path.join(self.data.root, self.data.imgs[index])
        elif isinstance(self.data, datasets.ImageFolder):
            path, _ = self.data.samples[index]
        else:
            raise NotImplementedError
        return img, target, index_, path

    def __len__(self):
        return len(self.data)

    def get_index(self, index_):
        if hasattr(self, 'index_map'):
            index_ = self.index_map[index_]
        return index_

    def get_item_path(self, index_):
        index = self.get_index(index_)

        if isinstance(self.data, utils.domainnet.Dataset):
            path = os.path.join(self.data.root, self.data.imgs[index])
        elif isinstance(self.data, datasets.ImageFolder):
            path, _ = self.data.samples[index]
        else:
            raise NotImplementedError
        return path
