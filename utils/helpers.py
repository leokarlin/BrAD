from PIL import Image
import requests
from io import BytesIO

import torchvision.transforms as transforms
from moco.loader import ToEdges

import skimage
import numpy as np

import torch
from tqdm import tqdm

from utils import domainnet
from sklearn.neighbors import NearestNeighbors
import os


def loadImageOrURL(img):
    try:
        img_pil = Image.open(img)
    except:
        response = requests.get(img)
        img_pil = Image.open(BytesIO(response.content))
    return img_pil

#TODO: remove getTransforms? (is't it always False?)
def getTransforms(edges_sigma, process_scale, no_cc, img2_is_sketch, edges_type='canny'):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_sz = 224 * process_scale
    trans = transforms.Compose(
        [transforms.Resize(int(img_sz * 256.0 / 224.0))] + ([transforms.CenterCrop(img_sz), ] if not no_cc else []) +
        [transforms.ToTensor(), normalize, ])

    if edges_type == 'hed':
        trans2edge = transforms.Compose([transforms.Resize(int(img_sz * 256.0 / 224.0))] + ([transforms.CenterCrop(img_sz),] if not no_cc else []) +
                                        [transforms.ToTensor()])
        if img2_is_sketch:
            trans2edge = transforms.Compose([trans2edge, transforms.Lambda(lambda x: 1.0 - x) ])
    else:

        edgeMap = ToEdges(sigma=edges_sigma)

        normalize_edges = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                               std=[0.229, 0.224, 0.225])

        trans2edge = transforms.Compose([transforms.Resize(int(img_sz * 256.0 / 224.0))] + ([transforms.CenterCrop(img_sz),] if not no_cc else []) +
                                        [transforms.ToTensor(),transforms.Lambda(lambda x: 1.0 - x) if img2_is_sketch else edgeMap,normalize_edges])

    return trans, trans2edge


#input is a PIL image
def get_skeleton(img):
    return Image.fromarray(255 - skimage.morphology.skeletonize(np.asarray(img.convert('LA')) < 100)).convert('RGB')


def entropy_loss(scores):
    return torch.mean(
        - torch.sum(
            torch.nn.functional.log_softmax(scores, dim=1) *
            torch.nn.functional.softmax(scores, dim=1),
            dim=1
        ),
        dim=0
    )


def prepDomains(src_domains, model, trans, trans2edge, args):
    device = next(model.parameters()).device

    # pre-process the domains
    prep = []
    for src_domain in src_domains:
        sv_path = f'{src_domain}_features.npz'
        if not os.path.isfile(sv_path):
            train_dataset = domainnet.Dataset(
                args.data + f'/{src_domain}_train.txt',
                root=args.data,
                transform=trans,
                n_samples_per_class=args.n_samples_per_class
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

            if args.mlp:
                features_dim = 128
            else:
                features_dim = 2048
            if args.max_train_samples == -1:
                num_train_samples = len(train_dataset)
            else:
                num_train_samples = min(len(train_dataset), args.max_train_samples)

            train_features = np.zeros((num_train_samples, features_dim), dtype=np.float16)
            train_labels = np.zeros(num_train_samples, dtype=np.int64)
            train_images = train_dataset.imgs
            # for i, (xs, y) in tqdm(enumerate(train_loader)):
            # for x in xs:
            for i, (x, y) in tqdm(enumerate(train_loader)):
                with torch.no_grad():

                    logits, _, features = model(x.to(device))
                    if args.mlp:
                        features = logits
                    else:
                        features = torch.mean(features, dim=[-2, -1])
                    features = torch.nn.functional.normalize(features, dim=1)
                features = features.cpu().numpy().astype(np.float16)

                begin_ind = i * args.batch_size
                end_ind = min(begin_ind + len(features), len(train_features))
                train_features[begin_ind:end_ind, :] = features[:end_ind - begin_ind]
                train_labels[begin_ind:end_ind] = y[:end_ind - begin_ind]
                if end_ind == num_train_samples:
                    break
            np.savez(sv_path, train_features, train_labels, train_images)
        else:
            data = np.load(sv_path)
            train_features, train_labels, train_images = data['arr_0'], data['arr_1'], data['arr_2']
            data.close()
        print(
            f'train_features: {train_features.shape}, train_labels: {train_labels.shape}, train_images: {len(train_images)}')
        NN = NearestNeighbors(n_neighbors=20, algorithm='auto', n_jobs=-1).fit(train_features)
        prep.append([NN, [os.path.join(args.data, x) for x in train_images], train_features, train_labels])
    dmMap = {x: ix for ix, x in enumerate(src_domains)}
    prep = {'prep' : prep, 'dmMap' : dmMap, 'src_domains' : src_domains}
    return prep