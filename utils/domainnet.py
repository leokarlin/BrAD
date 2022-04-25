import numpy as np
import os
import os.path
from PIL import Image
import json
# subset of classes used in https://arxiv.org/abs/2103.16765
use_classes = (
    "aircraft_carrier",
    "alarm_clock",
    "ant",
    "anvil",
    "asparagus",
    "axe",
    "banana",
    "basket",
    "bathtub",
    "bear",
    "bee",
    "bird",
    "blackberry",
    "blueberry",
    "bottlecap",
    "broccoli",
    "bus",
    "butterfly",
    "cactus",
    "cake",
    "calculator",
    "camel",
    "camera",
    "candle",
    "cannon",
    "canoe",
    "carrot",
    "castle",
    "cat",
    "ceiling_fan",
    "cello",
    "cell_phone",
    "chair",
    "chandelier",
    "coffee_cup",
    "compass",
    "computer",
    "cow",
    "crab",
    "crocodile",
    "cruise_ship",
    "dog",
    "dolphin",
    "dragon",
    "drums",
    "duck",
    "dumbbell",
    "elephant",
    "eyeglasses",
    "feather",
    "fence",
    "fish",
    "flamingo",
    "flower",
    "foot",
    "fork",
    "frog",
    "giraffe",
    "goatee",
    "grapes",
    "guitar",
    "hammer",
    "helicopter",
    "helmet",
    "horse",
    "kangaroo",
    "lantern",
    "laptop",
    "leaf",
    "lion",
    "lipstick",
    "lobster",
    "microphone",
    "monkey",
    "mosquito",
    "mouse",
    "mug",
    "mushroom",
    "onion",
    "panda",
    "peanut",
    "pear",
    "peas",
    "pencil",
    "penguin",
    "pig",
    "pillow",
    "pineapple",
    "potato",
    "power_outlet",
    "purse",
    "rabbit",
    "raccoon",
    "rhinoceros",
    "rifle",
    "saxophone",
    "screwdriver",
    "sea_turtle",
    "see_saw",
    "sheep",
    "shoe",
    "skateboard",
    "snake",
    "speedboat",
    "spider",
    "squirrel",
    "strawberry",
    "streetlight",
    "string_bean",
    "submarine",
    "swan",
    "table",
    "teapot",
    "teddy-bear",
    "television",
    "The_Eiffel_Tower",
    "The_Great_Wall_of_China",
    "tiger",
    "toe",
    "train",
    "truck",
    "umbrella",
    "vase",
    "watermelon",
    "whale",
    "zebra"
)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list, filter_files=(), n_samples_per_class=-1):
    print(f'image_list:{image_list}')
    with open(image_list) as f:
        if image_list[-4:] == 'json':
            jdata = json.load(f)
            images_labels = [(x['photo'], x['product']) for x in jdata]
            bboxes = [(x['bbox']['left'], x['bbox']['top'], x['bbox']['width'], x['bbox']['height']) for x in jdata]
        else:
            images_labels = [x.strip().split(' ') for x in f.readlines()]
            bboxes = []
    if filter_files:
        images_labels = [(img, label) for (img, label) in images_labels if any([f in img for f in filter_files])]
    images = np.array([x for (x, y) in images_labels])
    labels = np.array([int(y) for (x, y) in images_labels])
    if n_samples_per_class > 0:
        chosen_images, chosen_labels = [], []
        for _ in range(n_samples_per_class):
            _, indices = np.unique(labels, return_index=True)
            chosen_images.append(images[indices])
            chosen_labels.append(labels[indices])
            images = np.delete(images, indices)
            labels = np.delete(labels, indices)
        images = np.concatenate(chosen_images)
        labels = np.concatenate(chosen_labels)

    return images, labels, bboxes


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


class Dataset(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False, filter_files=use_classes, n_samples_per_class=-1,
                 return_path=False):
        self.torch_dataset = False
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test
        self.return_path = return_path
        self.imgs, self.labels, self.bboxes = make_dataset_fromlist(image_list, filter_files, n_samples_per_class)
        self.index_map = np.arange(len(self.imgs))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if len(self.bboxes) > 0:
            bbox = self.bboxes[index]
            img = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:

            return img, target, self.imgs[index]

    def __len__(self):

        return len(self.imgs)
