#!/usr/bin/env python
import math
import os
import random
import shutil
import time
import warnings
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils.torchvision_wrappers as models_wrappers
from utils import domainnet
from utils.data import DataCoLoader, IndexedDataset
import moco.loader
from config import parser, setup
from moco.builder import concat_all_gather

# from cvar_pyutils.debugging_tools import set_remote_debugger # TODO: Remove Debugger


def main(args=None):
    if args is None:
        args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        if not args.debug:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')
    init_ddp(args.local_rank)

    main_worker(args.local_rank, args)


def init_ddp(local_rank):
    if 'RANK' not in os.environ.keys():
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ.keys():
        os.environ['LOCAL_RANK'] = str(local_rank)
    if 'WORLD_SIZE' not in os.environ.keys():
        os.environ['WORLD_SIZE'] = '1'
    if 'MASTER_PORT' not in os.environ.keys():
        os.environ['MASTER_PORT'] = '55555'
    if 'MASTER_ADDR' not in os.environ.keys():
        os.environ['MASTER_ADDR'] = 'localhost'

    torch.distributed.init_process_group('nccl')


def main_worker(gpu, args):
    args.gpu = gpu
    args = setup(args)  # TODO: Go over setup

    #if args.is_root and args.debug:
    #    set_remote_debugger(debug_port=12345)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model
    print("=> creating model '{}'".format(args.arch))
    import moco.builder
    model = moco.builder.MoCo(
        models_wrappers.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp,
        debug=args.debug,
        args=args
    )

    if args.edges_type == 'hed':
        enet = []
        for _ in args.data.split(','):
            enet.append(moco.builder.HedNet(args, pretrained_hed='canny' not in args.hed_loss_type))
        if args.train_hed:
                if  args.hed_loss_type == 'l2_hed':
                    enet_orig = moco.builder.HedNet(args, train_hed=False, pretrained_hed=True)
                elif 'canny' in args.hed_loss_type:
                    enet_orig = moco.loader.ToEdges(sigma=args.edges_sigma, dil=args.edges_dil)
        else:
            enet_orig = None
    else:
        enet = None
        enet_orig = None

    ddisc = moco.builder.DomainDiscriminator(args) if args.ddisc else None

    print(f'=> original args.batch_size={args.batch_size}')

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / dist.get_world_size())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    if args.ddisc:
        ddisc.cuda(args.gpu)
        ddisc = torch.nn.parallel.DistributedDataParallel(ddisc, device_ids=[args.gpu], output_device=args.gpu)

    if args.edges_type == 'hed':
        for ei, e in enumerate(enet):
            e.cuda(args.gpu)
            enet[ei] = torch.nn.parallel.DistributedDataParallel(e, device_ids=[args.gpu])
        if args.train_hed:
            enet_orig.cuda(args.gpu)

    print(f'=> modified args.batch_size={args.batch_size}')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer_ddisc = torch.optim.SGD(ddisc.parameters(), args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay) if args.ddisc else None

    if args.train_hed:  # in the future we may want a separate set of hyper-parameters for enet
        l1_criterion = nn.L1Loss().cuda(args.gpu)
        l2_criterion = nn.MSELoss().cuda(args.gpu)
        optimizer_enet = []
        for e in enet:
            optimizer_enet.append(torch.optim.SGD(e.parameters(), args.lr, momentum=args.momentum,
                                                  weight_decay=args.weight_decay))
    else:
        optimizer_enet = None
        l1_criterion = None
        l2_criterion = None
    # optionally resume from a checkpoint
    if args.resume:
        if not os.path.isfile(args.resume):
            if args.resume != 'not a path':
                print("=> no checkpoint found at '{}'".format(args.resume))
            if os.path.isfile(os.path.join(args.work_folder, 'checkpoint_last.pth.tar')):
                args.resume = os.path.join(args.work_folder, 'checkpoint_last.pth.tar')
            else:
                gpp = os.path.join(args.work_folder, 'checkpoint_*.pth.tar')
                print(f'looking for latest checkpoint using: {gpp}')
                cpts = glob.glob(gpp)
                cpts_ix = [int(x.split('/')[-1].split('_')[1].split('.')[0]) for x in cpts]
                if len(cpts_ix) > 0:
                    mx_ix = np.argmax(cpts_ix)
                    args.resume = cpts[mx_ix]
                else:
                    print('=> no checkpoints found')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            if args.debug:
                sd = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            else:
                sd = checkpoint['state_dict']
            missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
            if (len(missing_keys) > 0) or (len(unexpected_keys) > 0):
                print(f'=> missing_keys={missing_keys}, unexpected_keys={unexpected_keys}')
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.ddisc:
                ddisc_sd = checkpoint['ddisc_state_dict']
                ddisc.load_state_dict(ddisc_sd)
                optimizer_ddisc.load_state_dict(checkpoint['ddisc_optimizer'])
            if args.train_hed:
                enet_sd = checkpoint['enet_state_dict']

                for ei in range(len(enet)):
                    enet[ei].load_state_dict(enet_sd[ei])
                    optimizer_enet[ei].load_state_dict(checkpoint['enet_optimizer'][ei])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    if args.edges_type == 'hed':
        if args.train_hed and 'canny' in args.hed_loss_type:
            augmentation = (
                augmentation.copy(),  # for queries
                augmentation[:-1],  # for keys #removed normalization
                [moco.loader.ToEdges2D(sigma=args.edges_sigma, dil=args.edges_dil),
                 transforms.GaussianBlur(args.canny_blur_rad, 0.15), moco.loader.StretchValues()]
            )
        else:
            augmentation = (
                augmentation.copy(),  # for queries
                augmentation[:-1]  # for keys #removed normalization
            )
    elif args.edges_type == 'canny':
        canny_map = moco.loader.ToEdges(sigma=args.edges_sigma, dil=args.edges_dil)
        augmentation = (
            # for queries
            augmentation.copy(),
            # for keys
            augmentation[:-1] + [canny_map, transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225])]
        )
    else:
        raise NotImplementedError()
    if args.train_hed and 'canny' in args.hed_loss_type:
        transform = moco.loader.TwoCropsTransformPlusCanny((transforms.Compose(augmentation[0]),
                                                            transforms.Compose(augmentation[1]),
                                                            transforms.Compose(augmentation[2])), args)
    else:
        transform = moco.loader.TwoCropsTransform(
            (transforms.Compose(augmentation[0]), transforms.Compose(augmentation[1])),
            args)

    train_loaders = []
    train_samplers = []
    datas = args.data.split(',')
    assert args.batch_size % len(datas) == 0, f'for simplicity, please make sure args.batch_size={args.batch_size} ' \
                                              f'is divisible by len(datasets)={len(datas)}'
    for iData, data in enumerate(datas):
        if os.path.isdir(data):
            print(f'=> Loading a custom dataset: {data}')
            train_dataset = datasets.ImageFolder(data, transform)
            print(f'=> loaded {len(train_dataset)} images')
        else:
            train_dataset = domainnet.Dataset(data, root=os.path.dirname(data), transform=transform, filter_files=[])

        # we wrap this to get the indices and paths for the batch out
        train_dataset = IndexedDataset(train_dataset, args)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_samplers.append(train_sampler)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=int(args.batch_size / len(datas)), shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

        train_loaders.append(train_loader)


    if len(train_loaders) > 1:
        train_loader = DataCoLoader(train_loaders, args)
    else:
        train_loader = train_loaders[0]

    print(f'=> Starting training from epoch {args.start_epoch}, will train for {args.epochs} epochs')

    for epoch in range(args.start_epoch, args.epochs):
        for train_sampler in train_samplers:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        if args.ddisc:
            adjust_learning_rate(optimizer_ddisc, epoch, args)
        if args.train_hed:
            for ei in range(len(enet)):
                adjust_learning_rate(optimizer_enet[ei], epoch, args)
        # train for one epoch
        print(f'=> Start training epoch #{epoch}')
        train(train_loader, model, ddisc, enet, enet_orig, criterion, optimizer, optimizer_ddisc, optimizer_enet,
              l1_criterion, l2_criterion, epoch, args)
        print(f'=> Finished training epoch #{epoch}')
        checkpoint_freq = args.save_n_epochs
        if args.is_root:
            save_dict = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'ddisc_state_dict': ddisc.state_dict() if args.ddisc else {},
                'optimizer': optimizer.state_dict(),
                'enet_state_dict': [],
                'enet_optimizer': [],
                'ddisc_optimizer': optimizer_ddisc.state_dict() if args.ddisc else {},
            }

            for ei in range(len(enet)):
                save_dict['enet_state_dict'].append(enet[ei].state_dict())
                save_dict['enet_optimizer'].append(optimizer_enet[ei].state_dict())
            save_checkpoint(save_dict, is_best=False, filename=os.path.join(args.work_folder,
                                                                            'checkpoint_last.pth.tar'))
            if (epoch+1) % checkpoint_freq == 0:
                src = os.path.join(args.work_folder, 'checkpoint_last.pth.tar')
                dst = os.path.join(args.work_folder, 'checkpoint_{:04d}.pth.tar'.format(epoch))
                shutil.copy(src, dst)


def train(train_loader, model, ddisc, enet, enet_orig, criterion, optimizer, optimizer_ddisc, optimizer_enet,
          l1_criterion, hed_criterion, epoch, args):

    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # When using trainable hed:
    losses_hed_meter = AverageMeter('Hed Loss', ':.4e')
    # When using trainable domain discriminator:
    losses_ddisc = AverageMeter('DDisc Loss', ':.4e')
    losses_ddisc_main_obj = AverageMeter('Obj. DDisc Loss', ':.4e')
    top1_ddisc_single = AverageMeter(f'DDisc Acc@image (t={args.ddisc_acc_enabler})', ':6.2f') # TODO: merge with ddisc_grid?
    top1_ddisc_grid = AverageMeter(f'DDisc Acc@grid '
                                   f'(t={args.ddisc_acc_enabler})', ':6.2f')
    meters = [batch_time, data_time, losses, top1, top5]
    if args.ddisc:
        meters.extend([losses_ddisc, losses_ddisc_main_obj, top1_ddisc_single, top1_ddisc_grid])
    if args.train_hed:
        meters.append(losses_hed_meter)
    progress = ProgressMeter(
        len(train_loader), meters, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    if args.ddisc:
        ddisc.train()
    if args.train_hed:

        for ei in range(len(enet)):
            enet[ei].train()

        if args.hed_loss_type == 'l2_hed':
            enet_orig.train()

    device = next(model.parameters()).device

    end = time.time()
    print(f'=> Entering train loop')

    for i, batch in enumerate(train_loader):
        images = None
        domain_label = []
        domain_index = []
        img_src = []
        for ib, b in enumerate(batch):
            b, _, di, isrc = b  # throw away the class labels
            if images is None:
                images = [[] for _ in range(len(b))]
            for j in range(len(b)):
                images[j].append(b[j])
            domain_label.append(ib * torch.ones((b[0].shape[0], 2), dtype=torch.long))
            domain_index.append(di.unsqueeze(dim=1).repeat(1, 2))
            img_src += isrc
        images = [torch.cat(img, dim=0) for img in images]
        # noinspection PyTypeChecker
        domain_label = torch.cat(domain_label, dim=0)
        domain_label = domain_label.view([-1] + list(domain_label.shape[2:])).to(device)
        domain_index = torch.cat(domain_index, dim=0)
        domain_index = domain_index.view([-1] + list(domain_label.shape[2:])).to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        images = [x.view([-1] + list(x.shape[2:])) for x in images]

        # metadata
        isedge_q, isedge_k = images[2], images[3]
        isedge_q_s = isedge_q.squeeze()
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        if args.train_hed:
            if 'canny' in args.hed_loss_type:
                orig_hed_q = images[4].squeeze(dim=1).cuda(args.gpu, non_blocking=True)

            elif args.hed_loss_type == 'l2_hed':

                # copmute original Hed results
                orig_hed_q_out, _ = enet_orig(im_q=images[0], im_k=images[1], isedge_q=isedge_q, isedge_k=isedge_k)
                orig_hed_q_out = orig_hed_q_out.detach()
                orig_hed_q = enet_orig.denormalize(orig_hed_q_out, isedge_q_s)
            else:
                raise NotImplementedError()

        # compute output
        if args.edges_type == 'hed':
            num_domains = int(domain_label[-1]+1)
            n_per_domain = int(len(domain_label)/num_domains)
            im_q = images[0]
            im_k = images[1]
            for n in range(num_domains):
                strt, stp = n*n_per_domain, (n+1)*n_per_domain
                im_q[strt:stp], im_k[strt:stp] = enet[n](im_q=im_q[strt:stp], im_k=im_k[strt:stp],
                                                         isedge_q=isedge_q[strt:stp],
                                                         isedge_k=isedge_k[strt:stp])

        else:
            im_q = images[0]
            im_k = images[1]
        output, target, extra_outputs = model(im_q=im_q, im_k=im_k, isedge_q=isedge_q, isedge_k=isedge_k,
                                              q_selector=domain_label, sample_idx=domain_index, sample_pth=img_src)
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # Domain Discriminator
        if args.ddisc:
            # prepare the target
            if not args.ddisc_images:
                ixv = isedge_q.squeeze().bool()
                ddisc_target = domain_label[ixv]
            else:
                ddisc_target = domain_label
                ixv = None

            # compute the domain discriminator loss
            loss_ddisc = 0
            loss_ddisc_main_obj = 0
            # Run domain discriminator on pooled features
            q = extra_outputs['q']  # n, c
            if not args.ddisc_images:
                q = q[ixv]
            ddisc_logits = ddisc(q)
            ddisc_logits_d = ddisc(q.detach())

            # gather ddisc_logits and ddisc_target
            if not args.debug:
                _ddisc_logits = concat_all_gather(ddisc_logits)
                _ddisc_target = concat_all_gather(ddisc_target)
            else:
                _ddisc_logits = ddisc_logits
                _ddisc_target = ddisc_target

            # accuracy handling
            ddisc_acc, _ = accuracy(_ddisc_logits, _ddisc_target, topk=(1, 1))
            top1_ddisc_single.update(ddisc_acc[0], images[0].size(0))

            crit_val = criterion(ddisc_logits, ddisc_target)
            crit_val_d = criterion(ddisc_logits_d, ddisc_target)
            if ddisc_acc[0] >= args.ddisc_acc_enabler:
                loss_ddisc_main_obj = loss_ddisc_main_obj + crit_val
            loss_ddisc = loss_ddisc + crit_val_d
            # Run domain discriminator on pre-pooled features
            q_fm = extra_outputs['q_fm']  # n, c, y, x
            if not args.ddisc_images:
                q_fm = q_fm[ixv]
            n, c, y, x = q_fm.shape
            q_fm = q_fm.permute(0, 2, 3, 1)
            ddisc_target_fm = ddisc_target.unsqueeze(1).unsqueeze(1).repeat(1, y, x)
            q_fm = q_fm.reshape(n * y * x, c)
            ddisc_target_fm = ddisc_target_fm.view(-1)
            ddisc_logits_fm = ddisc(q_fm)
            ddisc_logits_fm_d = ddisc(q_fm.detach())

            # gather ddisc_logits_fm and ddisc_target_fm
            if not args.debug:
                _ddisc_logits_fm = concat_all_gather(ddisc_logits_fm)
                _ddisc_target_fm = concat_all_gather(ddisc_target_fm)
            else:
                _ddisc_logits_fm = ddisc_logits_fm
                _ddisc_target_fm = ddisc_target_fm

            # accuracy handling
            ddisc_acc_fm, _ = accuracy(_ddisc_logits_fm, _ddisc_target_fm, topk=(1, 1))
            top1_ddisc_grid.update(ddisc_acc_fm[0], images[0].size(0))

            crit_val = criterion(ddisc_logits_fm, ddisc_target_fm)
            crit_val_d = criterion(ddisc_logits_fm_d, ddisc_target_fm)
            if ddisc_acc_fm[0] >= args.ddisc_acc_enabler:
                loss_ddisc_main_obj = loss_ddisc_main_obj + crit_val
            loss_ddisc = loss_ddisc + crit_val_d

            # update averages
            losses_ddisc.update(loss_ddisc.item() if isinstance(loss_ddisc, torch.Tensor)
                                else loss_ddisc, images[0].size(0))
            losses_ddisc_main_obj.update(loss_ddisc_main_obj.item() if isinstance(loss_ddisc_main_obj, torch.Tensor)
                                         else loss_ddisc_main_obj, images[0].size(0))

            # add the negative loss to the main objective
            loss = loss - loss_ddisc_main_obj

            # train the domain discriminator
            optimizer_ddisc.zero_grad()
            loss_ddisc.backward()

        if args.train_hed:
            # compute train Hed loss
            if args.hed_loss_type == 'l2_hed':
                hed_loss = hed_criterion(enet[0].module.denormalize(im_q, isedge_q_s), orig_hed_q)
            elif args.hed_loss_type == 'l1_canny':
                curr_edges = enet[0].module.denormalize(im_q, isedge_q_s)
                hed_loss = l1_criterion(curr_edges, orig_hed_q)
            elif args.hed_loss_type == 'l2_canny':
                curr_edges = enet[0].module.denormalize(im_q, isedge_q_s)
                hed_loss = hed_criterion(curr_edges, orig_hed_q)
            else:
                raise NotImplementedError()
            loss = loss + args.hed_loss_w * hed_loss
            losses_hed_meter.update(hed_loss)

            for ei in range(len(optimizer_enet)):
                optimizer_enet[ei].zero_grad()
        # compute gradient and do SGD step, if having a domain discriminator we lock its params from computing
        # gradients yet another time
        optimizer.zero_grad()
        if args.ddisc:
            ddisc.params_lock() if not isinstance(ddisc, torch.nn.parallel.DistributedDataParallel) \
                else ddisc.module.params_lock()
        loss.backward()

        if args.ddisc:
            ddisc.params_unlock() if not isinstance(ddisc, torch.nn.parallel.DistributedDataParallel) \
                else ddisc.module.params_unlock()
        optimizer.step()

        # now we can update the domain discriminator
        if args.ddisc:
            optimizer_ddisc.step()

        if args.train_hed:
            for ei in range(len(optimizer_enet)):
                optimizer_enet[ei].step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = self.avg = self.sum = self.count = 0

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


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []

        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            if batch_size > 0:
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                res.append(correct_k.mul_(0.0))
        return res


if __name__ == '__main__':

    main()
