import pickle
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import warnings
import random
import numpy as np
import time
import os
from utils.helpers import getTransforms
from main_brad import init_ddp
import moco.builder as mb
from config import parser, setup
from utils import domainnet
import torch.nn.functional as func
import torch.utils.data
from tqdm import tqdm

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import NearestNeighbors
import torch.distributed as dist


def get_argparse():
    parser.add_argument('--src-domain', type=str, help='path to file with source domain image paths and classes')
    parser.add_argument('--dst-domain', type=str, help='path to file with target domain image paths and classes')
    parser.add_argument('--src_save_pth', type=str, default='',
                        help='path for saving source features. If empty nothing will be saved')
    parser.add_argument('--dst_save_pth', type=str, default='',
                        help='path for saving target features. If empty nothing will be saved')
    parser.add_argument('--classifier', default='retrieval', choices=["retrieval", "sgd", "logistic"], help='{"retrieval", "sgd", "logistic"}')

    return parser


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def cls_metrics(test_features, test_labels, cls):
    acc1 = cls.score(test_features, test_labels)
    return acc1, 0, 0, 0, 0, 0, 0, 0


def ret_metrics(test_features, test_labels, train_labels, nearest_neigh):
    _, top_n_matches_ids = nearest_neigh.kneighbors(test_features)
    top_n_labels = train_labels[top_n_matches_ids]
    correct = test_labels[:, None] == top_n_labels

    acc1 = correct[:, 0:1].any(-1).mean()
    acc10 = correct[:, 0:10].any(-1).mean()
    acc20 = correct[:, 0:20].any(-1).mean()

    p1 = correct[:, 0:1].sum(-1).mean()
    p10 = correct[:, 0:10].sum(-1).mean() / 10
    p20 = correct[:, 0:20].sum(-1).mean() / 20
    p5 = correct[:, 0:5].sum(-1).mean() / 5
    p15 = correct[:, 0:15].sum(-1).mean() / 15
    return acc1, acc10, acc20, p1, p5, p10, p15, p20


def main(args):
    init_ddp(args.local_rank)
    args.gpu = args.local_rank
    print(dist.get_rank())
    is_root = dist.get_rank() == 0
    torch.cuda.set_device(args.local_rank)
    # setup stuff
    args = setup(args)

    # seeding the random if required
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to fix the CUDNN random seed. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow you down considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # setup transforms
    trans, _ = getTransforms(args.edges_sigma, args.process_scale, args.no_cc, args.img2_is_sketch, args.edges_type)

    # loading model
    model, cur_epoch = mb.load_model(args, return_epoch=args.resume)
    device = next(model.parameters()).device
    model.isedge = False
    features_dim = 2048

    # loading source domain data
    print(f"Source domain {args.src_domain}")
    if os.path.exists(args.src_save_pth):
        with open(args.src_save_pth, 'rb') as fp:
            img_paths, train_features, train_labels = pickle.load(fp)
    else:
        file_list_path = args.src_domain
        train_dataset = domainnet.Dataset(
            file_list_path,
            root=os.path.split(file_list_path)[0],
            test=True,
            transform=trans,
            n_samples_per_class=-1
        )
        ddp_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            sampler=ddp_sampler,
            num_workers=args.workers, pin_memory=True, drop_last=False)
        num_train_samples = len(train_dataset)
        train_features = np.zeros((num_train_samples, features_dim), dtype=np.float16)
        train_labels = np.zeros(num_train_samples, dtype=np.int64)
        img_paths = []
        end_ind = 0
        for ind, (x, y, img_path) in tqdm(enumerate(train_loader)):
            with torch.no_grad():
                _, _, features = model(x.to(device))
                features = torch.mean(features, dim=[-2, -1])
                features = nn.functional.normalize(features, dim=1)
                features = mb.concat_all_gather_interlace(features)
                y = mb.concat_all_gather_interlace(y.to(device))
                img_path = mb.concat_all_gather_object_interlace(img_path)
                img_paths.extend(img_path)
                if not is_root:
                    continue
            y = y.cpu().numpy().astype(np.float16)
            features = features.cpu().numpy().astype(np.float16)
            begin_ind = end_ind
            end_ind = min(begin_ind + len(features), len(train_features))
            train_features[begin_ind:end_ind, :] = features[:end_ind - begin_ind]
            train_labels[begin_ind:end_ind] = y[:end_ind - begin_ind]
            if end_ind >= num_train_samples:
                break
        if is_root:
            if args.src_save_pth:
                if os.path.split(args.src_save_pth)[0]:
                    os.makedirs(os.path.split(args.src_save_pth)[0], exist_ok=True)
                with open(args.src_save_pth, 'wb') as fp:
                    pickle.dump((img_paths, train_features, train_labels), fp, pickle.HIGHEST_PROTOCOL)
    if is_root:
        if args.classifier == 'sgd':
            cls = SGDClassifier(max_iter=1000, n_jobs=16, tol=1e-3).fit(train_features, train_labels)
        elif args.classifier == 'logistic':
            cls = LogisticRegression(max_iter=1000, n_jobs=16, tol=1e-3).fit(train_features, train_labels)
        elif args.classifier == 'retrieval':
            cls = NearestNeighbors(n_neighbors=min(20, train_features.shape[0]), algorithm='auto',
                                   n_jobs=-1, metric='correlation').fit(train_features)
        else:
            raise NotImplementedError()
    else:
        cls = None

    print(f"Target domain {args.dst_domain}")

    batch_time = AverageMeter('Time', ':6.5f')
    acc1 = AverageMeter('Acc@1', ':6.5f')
    acc10 = AverageMeter('Acc@10', ':6.5f')
    acc20 = AverageMeter('Acc@20', ':6.5f')
    precision1 = AverageMeter('p@1', ':6.5f')
    precision10 = AverageMeter('p@10', ':6.5f')
    precision20 = AverageMeter('p@20', ':6.5f')
    precision5 = AverageMeter('p@5', ':6.3f')
    precision15 = AverageMeter('p@15', ':6.3f')

    if os.path.exists(args.dst_save_pth):
        with open(args.dst_save_pth, 'rb') as fp:
            saved_features = pickle.load(fp)

            if args.is_root:
                if args.classifier == 'retrieval':
                    a1, a10, a20, p1, p5, p10, p15, p20 = ret_metrics(saved_features[1], saved_features[2],
                                                                      train_labels, cls)
                else:
                    a1, a10, a20, p1, p5, p10, p15, p20 = cls_metrics(saved_features[1], saved_features[2],
                                                                      cls)
            else:
                a1, a10, a20, p1, p5, p10, p15, p20 = 0., 0., 0., 0., 0., 0., 0., 0.
            y = saved_features[2]
            acc1.update(a1, len(y))
            acc10.update(a10, len(y))
            acc20.update(a20, len(y))
            precision1.update(p1, len(y))
            precision5.update(p5, len(y))
            precision10.update(p10, len(y))
            precision15.update(p15, len(y))
            precision20.update(p20, len(y))
    else:
        file_list_path = args.dst_domain
        test_dataset = domainnet.Dataset(
            file_list_path,
            test=True,
            root=os.path.split(file_list_path)[0],
            transform=trans,
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, sampler=test_sampler,
            num_workers=args.workers, pin_memory=True)
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, acc1, acc10, acc20, precision1, precision5, precision10, precision15, precision20],
            prefix=f"Train on {os.path.basename(args.src_domain)} Test on {os.path.basename(args.dst_domain)}")
        end = time.time()
        num_test_samples = len(test_dataset)
        print(f'num_test_samples: {num_test_samples}')
        total_samples = 0
        img_paths = []
        all_features = []
        all_y = []
        for ind, (x, y, img_path) in enumerate(test_loader):
            with torch.no_grad():
                _, _, features = model(x.to(device))
                features = torch.mean(features, dim=[-2, -1])
                features = nn.functional.normalize(features, dim=1)
                features = mb.concat_all_gather_interlace(features)
                y = mb.concat_all_gather_interlace(y.to(device)).cpu()
                img_path = mb.concat_all_gather_object_interlace(img_path)
                img_paths.extend(img_path)
                if not is_root:
                    continue
            features = features.cpu().numpy().astype(np.float16)
            y = y.numpy()
            total_samples += len(y)
            if total_samples > num_test_samples:
                diff = total_samples - num_test_samples
                features = features[:-diff]
                y = y[:-diff]
                all_features.append(features)
                all_y.append(y)
                print(f'diff {diff}')
            all_features.append(features)
            all_y.append(y)
            if args.classifier == 'retrieval':
                a1, a10, a20, p1, p5, p10, p15, p20 = ret_metrics(features, y, train_labels, cls)
            else:
                a1, a10, a20, p1, p5, p10, p15, p20 = cls_metrics(features, y, cls)
            acc1.update(a1, len(y))
            acc10.update(a10, len(y))
            acc20.update(a20, len(y))
            precision1.update(p1, len(y))
            precision10.update(p10, len(y))
            precision20.update(p20, len(y))
            precision5.update(p5, len(y))
            precision15.update(p15, len(y))
            batch_time.update(time.time() - end)
            end = time.time()

            if ind % 10 == 0:
                progress.display(ind)

        if args.dst_save_pth:
            if os.path.split(args.dst_save_pth)[0]:
                os.makedirs(os.path.split(args.dst_save_pth)[0], exist_ok=True)
            with open(args.dst_save_pth, 'wb') as fp:
                pickle.dump((img_paths, np.concatenate(all_features), np.concatenate(all_y)),
                            fp, pickle.HIGHEST_PROTOCOL)

        print('Final batch:')
        progress.display(len(test_loader))
    if args.is_root:
        print('Results:')
        print(','.join([
                          f'acc1={acc1.avg}',
                          f'acc10={acc10.avg}',
                          f'acc20={acc20.avg}',
                          f'precision1={precision1.avg}',
                          f'precision5={precision5.avg}',
                          f'precision10={precision10.avg}',
                          f'precision15={precision15.avg}',
                          f'precision20={precision20.avg}'
                        ]))


if __name__ == '__main__':
    parser = get_argparse()
    main(parser.parse_args())


