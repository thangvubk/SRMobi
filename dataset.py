import glob
import os.path as osp
import random

import imageio
import numpy as np


class Dataset(object):

    def __init__(self,
                 root='./data',
                 dataset='DIV2K',
                 split='train',
                 scale=4,
                 crop_cfg=dict(type='random', patch_size=48)):
        assert dataset in ['DIV2K']
        assert split in ['train', 'val', 'test']
        self.root = root
        self.dataset = dataset
        self.split = split
        self.scale = scale
        self.crop_cfg = crop_cfg

        lr_path = osp.join(root, split, dataset, 'LR')
        hr_path = osp.join(root, split, dataset, 'HR')
        self.lr_images = sorted(glob.glob(osp.join(lr_path, '*.png')))
        self.hr_images = sorted(glob.glob(osp.join(hr_path, '*.png')))

        assert len(self.lr_images) > 0 and len(self.hr_images) > 0

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = self.lr_images[idx]
        hr_path = self.hr_images[idx]

        lr = imageio.imread(lr_path) / 255
        hr = imageio.imread(hr_path) / 255

        if self.crop_cfg is not None:
            lr, hr = self._crop(lr, hr, self.crop_cfg)

        return lr, hr

    def _crop(self, lr, hr, crop_cfg):
        crop_type = crop_cfg['type']
        patch_size = crop_cfg['patch_size']
        ih, iw, ic = lr.shape  # shape of original image

        lr_patch_size = patch_size
        hr_patch_size = lr_patch_size * self.scale

        if crop_type == 'random':
            # indexing lr patch
            h = random.randint(0, ih - lr_patch_size)
            w = random.randint(0, iw - lr_patch_size)

            # indexing hr patch
            H = h * self.scale
            W = w * self.scale

        elif crop_type == 'fixed':
            h, w, H, W = 0, 0, 0, 0

        lr = lr[h:h + lr_patch_size, w:w + lr_patch_size, :]
        hr = hr[H:H + hr_patch_size, W:W + hr_patch_size, :]

        return lr, hr


class DataLoader(object):
    """ simple dataloader """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_len = len(dataset)
        self._reset_sampler()

    def _reset_sampler(self):
        if self.shuffle:
            self.sampler = iter(range(self.data_len))
        else:
            self.sampler = iter(np.random.permutation(self.data_len))

    def __iter__(self):
        self.iter = 0
        self._reset_sampler()
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > len(self):
            raise StopIteration
        batch_data = []
        for _ in range(self.batch_size):
            i = next(self.sampler)
            batch_data.append(self.dataset[i])
        return self._collate(batch_data)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _collate(self, batch_data):
        # convert list of tuple to tuple of list
        batch_data = tuple(map(list, zip(*batch_data)))
        collate_data = tuple(map(lambda x: np.stack(x, axis=0), batch_data))
        return collate_data


if __name__ == '__main__':
    dataset = Dataset()
    dataloader = DataLoader(dataset, 2)
    for epoch in range(2):
        for i, data in enumerate(dataloader):
            print(i)
