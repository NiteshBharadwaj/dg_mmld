from torch.utils.data import Dataset
import sys
import os
from torchvision import transforms
from torchvision.datasets.folder import make_dataset, default_loader
import numpy as np
from svhn import load_svhn
from mnist import load_mnist
from mnist_m import load_mnistm
from usps_ import load_usps
from gtsrb import load_gtsrb
from synth_number import load_syn
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
from dataloader.OfficeRead import read_office_domain

def return_dataset(data, scale=False, usps=False, all_use='no', directory="."):
    if data == 'svhn':
        train_image, train_label, \
        test_image, test_label = load_svhn(directory)
    if data == 'mnist':
        train_image, train_label, \
        test_image, test_label = load_mnist(directory)
        # print(train_image.shape)
    if data == 'mnistm':
        train_image, train_label, \
        test_image, test_label = load_mnistm(directory)
        # print(train_image.shape)
    if data == 'usps':
        train_image, train_label, \
        test_image, test_label = load_usps(directory)
    if data == 'gtsrb':
        train_image, train_label, \
        test_image, test_label = load_gtsrb(directory)
    if data == 'syn':
        train_image, train_label, \
        test_image, test_label = load_syn(directory)

    return train_image, train_label, test_image, test_label
from PIL import Image

class DG_Dataset(Dataset):
    def __init__(self, root_dir, domain, split, get_domain_label=False, get_cluster=False, color_jitter=True,
                 min_scale=0.8, seed=0):
        self.root_dir = root_dir
        self.domain = domain
        self.split = split
        self.get_domain_label = get_domain_label
        self.get_cluster = get_cluster
        self.color_jitter = color_jitter
        self.min_scale = min_scale
        self.set_transform(self.split)
        self.loader = default_loader
        self.num_class = 31
        self.seed = seed
        self.load_dataset()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        if img.shape[0] != 1:
            # print(img)
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # print(np.vstack([im,im,im]).shape)
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        image = self.transform(img)
        output = [image, target]

        if self.get_domain_label:
            domain = np.copy(self.domains[index])
            domain = np.int64(domain)
            output.append(domain)

        if self.get_cluster:
            cluster = np.copy(self.clusters[index])
            cluster = np.int64(cluster)
            output.append(cluster)
        else:
            output.append(0)
        return tuple(output)

    def find_classes(self, dir_name):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir_name) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def load_dataset(self):
        images_list = []
        labels_list = []
        domains_list = []
        d_id = 0
        for d in self.domain:
            train, train_label, test, test_label = return_dataset(d)
            indices_tar = np.arange(0, test.shape[0])
            np.random.seed(self.seed)
            np.random.shuffle(indices_tar)
            val_split = int(0.05 * test.shape[0])
            val = test[indices_tar[:val_split]]
            val_label = test[indices_tar[:val_split]]
            target_test = test[indices_tar[val_split:]]
            target_test_label = test_label[indices_tar[val_split:]]
            images = None
            labels = None
            if self.split == "train":
                images = train
                labels = train_label
            elif self.split == "val":
                images = val
                labels = val_label
            elif self.split == "test":
                images = target_test
                labels = target_test_label
            images_list = images_list.append(images)
            labels_list = labels_list.append(labels)
            domains_list = domains_list + (np.zeros(len(images)) + d_id).astype(int).tolist()

        self.images = np.concatenate(images_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)
        self.domains = domains_list
        self.clusters = np.zeros(len(self.domains), dtype=np.int64)

    def set_cluster(self, cluster_list):
        if len(cluster_list) != len(self.images):
            raise ValueError("The length of cluster_list must to be same as self.images")
        else:
            self.clusters = cluster_list
            self.get_cluster = True

    def set_domain(self, domain_list):
        if len(domain_list) != len(self.images):
            raise ValueError("The length of domain_list must to be same as self.images")
        else:
            self.domains = domain_list

    def set_transform(self, split):
        self.transform = transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
