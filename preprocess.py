import glob
import os
import os.path as osp

import imageio
import numpy as np

SRC_PATH = 'data/origin'
DEST_PATH = 'data/preprocess'
DATASET = 'DIV2K'
SPLIT = 'test'

print('Reading data...')
lr_path = osp.join(SRC_PATH, SPLIT, DATASET, 'LR')
hr_path = osp.join(SRC_PATH, SPLIT, DATASET, 'HR')
lr_globs = glob.glob(osp.join(lr_path, '*.png'))
hr_globs = glob.glob(osp.join(hr_path, '*.png'))
assert len(lr_globs) > 0 and len(hr_globs) > 0
hr_globs.sort()
lr_globs.sort()
lr_images = [imageio.imread(lr) for lr in lr_globs]
hr_images = [imageio.imread(hr) for hr in hr_globs]

print('Writing data to npy...')
bin_path = osp.join(DEST_PATH, SPLIT, DATASET)
if not osp.exists(bin_path):
    os.makedirs(bin_path)

np.save(os.path.join(bin_path, 'lr.npy'), lr_images)
np.save(os.path.join(bin_path, 'hr.npy'), hr_images)
print('Complete!')
