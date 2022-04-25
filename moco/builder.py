import itertools
import torch
import torch.nn as nn
import torch.nn.functional as func
import os
import glob
import numpy as np
import utils.torchvision_wrappers as models_wrappers
import lib.hed_pytorch.hed as hed
import torch.distributed as dist


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, debug=False, args=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.debug = debug
        self.args = args

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, pretrained=args.imagenet_pretrained)
        self.encoder_k = base_encoder(num_classes=dim, pretrained=args.imagenet_pretrained)

        if mlp:  # hack: brute-force replacement

            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.qMult = 1
        if self.args.multi_q:
            self.qMult = len(self.args.data.split(','))
            self.register_buffer("queue", torch.randn(self.qMult, dim, K))
            self.queue = nn.functional.normalize(self.queue, dim=1)
            self.register_buffer("queue_idx", - torch.ones(self.qMult, K))
        else:
            self.register_buffer("queue", torch.randn(dim, K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_idx", - torch.ones(K))

        self.Q_lim = [None] * self.qMult
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, q_selector=None, sample_idx=None):
        # gather keys before updating queue
        if not self.debug:
            keys = concat_all_gather(keys)
            if q_selector is not None:
                q_selector = concat_all_gather(q_selector)
            if sample_idx is not None:
                sample_idx = concat_all_gather(sample_idx)
        # also for simplicity, here we assume each domain contributes the same number of samples to the batch
        batch_size = keys.shape[0] // self.qMult

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, f'self.K={self.K}, batch_size={batch_size}'  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if q_selector is not None:
            for iQ in range(self.qMult):
                if self.Q_lim[iQ] is not None:
                    local_ptr = ptr % self.Q_lim[iQ]
                else:
                    local_ptr = ptr
                active_keys = keys[q_selector == iQ]

                self.queue[iQ, :, local_ptr:local_ptr + batch_size] = active_keys.T
                self.queue_idx[iQ, local_ptr:local_ptr + batch_size] = sample_idx[q_selector == iQ]
        else:
            if self.Q_lim[0] is not None:
                local_ptr = ptr % self.Q_lim[0]
            else:
                local_ptr = ptr

            self.queue[:, local_ptr:local_ptr + batch_size] = keys.T
            self.queue_idx[local_ptr:local_ptr + batch_size] = sample_idx

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, ix=None):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        if ix is not None:
            ix_gather = concat_all_gather(ix)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        dist.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        ix_ret = None
        if ix is not None:
            ix_ret = ix_gather[idx_this]

        return x_gather[idx_this], ix_ret, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, isedge_q=None, isedge_k=None, q_selector=None, sample_idx=None, sample_pth=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            isedge_q: binary index, 0 = image, 1 = edge
            isedge_k: binary index, 0 = image, 1 = edge
        Output:
            logits, targets
        """

        extra_outputs = {}

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC

        if isinstance(q, tuple) or isinstance(q, list):  # handle the case features are returned too
            extra_outputs['q_fm'] = nn.functional.normalize(q[2], dim=1)
            extra_outputs['q'] = nn.functional.normalize(torch.mean(torch.mean(q[2], dim=3), dim=2), dim=1)
            q = q[0]
        q = nn.functional.normalize(q, dim=1)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if not self.debug:
                im_k, isedge_k, idx_unshuffle = self._batch_shuffle_ddp(im_k, isedge_k)

            k = self.encoder_k(im_k)  # keys: NxC

            if isinstance(k, tuple) or isinstance(k, list):  # handle the case features are returned too
                k = k[0]
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if not self.debug:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle) # no need to unshuffle the "isedge_k" as they are no longer used beyond this point

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        BIG_NUMBER = 10000.0 * self.T  # taking temp into account for a good measure
        l_neg = (torch.zeros((q.shape[0], self.K)).cuda() - BIG_NUMBER)
        if self.args.multi_q:
            for iQ in range(self.qMult):
                if self.Q_lim[iQ] is not None:
                    qlim = self.Q_lim[iQ]
                else:
                    qlim = self.queue[iQ].shape[1]
                ixx = (q_selector == iQ)
                _l_neg = torch.einsum('nc,ck->nk', [q[ixx], self.queue[iQ][:,:qlim].clone().detach()])
                if sample_idx is not None:
                    for ii, indx in enumerate(sample_idx[ixx]):
                        _l_neg[ii, self.queue_idx[iQ][:qlim] == indx] = - BIG_NUMBER
                l_neg[ixx, :qlim] = _l_neg
        else:
            if self.Q_lim[0] is not None:
                qlim = self.Q_lim[0]
            else:
                qlim = self.queue.shape[1]
            _l_neg = torch.einsum('nc,ck->nk', [q, self.queue[:, :qlim].clone().detach()])
            if sample_idx is not None:
                for ii in range(q.shape[0]):
                    _l_neg[ii, self.queue_idx[:qlim] == sample_idx[ii]] = - BIG_NUMBER
            l_neg[:, :qlim] = _l_neg

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, q_selector if self.args.multi_q else None, sample_idx)

        return logits, labels, extra_outputs

    def params_lock(self):
        for p in self.parameters():
            p.requires_grad_b_ = p.requires_grad
            p.requires_grad = False

    def params_unlock(self):
        for p in self.parameters():
            p.requires_grad = p.requires_grad_b_


# edge detector net
class HedNet(nn.Module):
    def __init__(self, args=None, train_hed=None, pretrained_hed=True):
        super(HedNet, self).__init__()
        self.args = args
        if train_hed is None:
            train_hed = args.train_hed
        self.edgeMean = [0.5, 0.5, 0.5]
        self.edgeSTD = [0.229, 0.224, 0.225]
        self.edgeNet = hed.Network(pretrained_hed)
        if not train_hed:
            self.edgeNet.eval()
            for param_edge in self.edgeNet.parameters():
                param_edge.requires_grad = False  # not update by gradient

    def denormalize(self, im, isedge):
        return im[isedge > 0, 0] * self.edgeSTD[0] + self.edgeMean[0]

    def normalize(self, edge):
        return torch.cat([
            ((edge - self.edgeMean[0]) / self.edgeSTD[0]),
            ((edge - self.edgeMean[1]) / self.edgeSTD[1]),
            ((edge - self.edgeMean[2]) / self.edgeSTD[2])
        ], 1)

    def normalized_edges(self, im, isedge):
        im_out = im.clone()
        isedge = isedge.squeeze()
        edge = self.edgeNet(im_out[isedge > 0, :]).clamp(0.0, 1.0)
        if self.args.stretch_edge:
            edge -= edge.min(dim=1)[0].min(dim=1)[0][:, None, None]
            edge /= (edge.max(dim=1)[0].max(dim=1)[0][:, None, None] + 1e-14)
        # normalizing edge images
        im_out[isedge > 0, :] = self.normalize(edge)
        return im_out

    def forward(self, im_q, im_k, isedge_q, isedge_k):
        im_q_out = self.normalized_edges(im_q, isedge_q)
        im_k_out = self.normalized_edges(im_k, isedge_k)
        return im_q_out, im_k_out


class DomainDiscriminator(nn.Module):
    def __init__(self, args=None):
        super(DomainDiscriminator, self).__init__()
        self.args = args
        ddisc_layers = [int(x) for x in args.ddisc_layers.split(',')]
        layers = []
        prev_n_f = 2048  # Todo: Why hard coded 2048??
        for i_l, n_f in enumerate(ddisc_layers):
            if i_l != 0:
                layers.append(torch.nn.Dropout(p=0.25))
            layers.append(torch.nn.Linear(prev_n_f, n_f))
            layers.append(torch.nn.LeakyReLU())
            prev_n_f = n_f
        layers.append(torch.nn.Linear(prev_n_f, len(args.data.split(','))))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def params_lock(self):
        for p in self.parameters():
            p.requires_grad_b_ = p.requires_grad
            p.requires_grad = False

    def params_unlock(self):
        for p in self.parameters():
            p.requires_grad = p.requires_grad_b_


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def concat_all_gather_interlace(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    inds = [torch.arange(i, len(g)*len(tensors_gather), len(tensors_gather)) for i, g in enumerate(tensors_gather)]
    inds = torch.cat(inds, dim=0)
    sort_inds = torch.argsort(inds)
    return output[sort_inds]


def concat_all_gather_object_interlace(obj):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    obj_gather = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(obj_gather, obj)
    if len(obj_gather[0]) > 1:
        return list(filter(None, itertools.chain(*itertools.zip_longest(*obj_gather))))
    else:
        return obj_gather


def load_model(args, return_epoch=False):
    # create model
    print("=> creating model '{}'".format(args.arch))

    model = models_wrappers.__dict__[args.arch](num_classes=args.moco_dim, pretrained=args.imagenet_pretrained)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    # load from pre-trained, before DistributedDataParallel constructor
    checkpoint = {}
    if args.resume:
        if not os.path.isfile(args.resume):
            if args.resume != 'not a path':
                print("=> no checkpoint found at '{}'".format(args.resume))

            if os.path.isfile(os.path.join(args.work_folder, 'checkpoint_last.pth.tar')):
                args.resume = os.path.join(args.work_folder, 'checkpoint_last.pth.tar')
            else:
                gpp = os.path.join(args.work_folder, 'checkpoint_*.pth.tar')
                cpts = glob.glob(gpp)
                cpts_ix = [int(x.split('/')[-1].split('_')[1].split('.')[0]) for x in cpts]
                if len(cpts_ix) > 0:
                    mx_ix = np.argmax(cpts_ix)
                    args.resume = cpts[mx_ix]
                else:
                    print('=> no checkpoints found')

        path2load = args.resume

        if os.path.isfile(path2load):
            print("=> loading checkpoint '{}'".format(path2load))
            checkpoint = torch.load(path2load, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q'): # and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            if args.mlp:
                dim_mlp = model.fc.weight.shape[1]
                model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            print(f'=> missing keys: {msg.missing_keys}')

            print("=> loaded pre-trained model '{}'".format(path2load))
        else:
            print("=> no checkpoint found at '{}'".format(path2load))

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)


    model.eval()

    if return_epoch:
        return model, checkpoint.get('epoch', 0)
    else:
        return model, 0

