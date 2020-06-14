import os.path as osp

import imageio
import numpy as np
import tensorflow as tf
from config import get_args
from dataset import DataLoader, Dataset
from model import EDSR
from tqdm import tqdm
from utils import compute_psnr, mkdir_p


def main():
    args = get_args()
    mkdir_p(args.result_dir)

    test_dataset = Dataset(split='test', crop_cfg=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = EDSR(num_blocks=args.num_blocks, channels=args.num_channels)
    assert args.checkpoint != '', 'checkpoint need to be specified'
    model.load_weights(args.checkpoint)

    # test
    print('Evaluating...')
    psnrs = []
    for i, (lr, hr) in enumerate(tqdm(test_loader)):
        lr = tf.constant(lr, dtype=tf.float32)
        hr = tf.constant(hr, dtype=tf.float32)
        sr = model(lr)
        # sr = tf.round(tf.clip_by_value(sr, 0, 255))
        psnr = compute_psnr(sr, hr)
        psnrs.append(psnr)
        if args.write_results:
            lr = lr.numpy()[0].clip(0, 255).round().astype(np.uint8)
            hr = hr.numpy()[0].clip(0, 255).round().astype(np.uint8)
            sr = sr.numpy()[0].clip(0, 255).round().astype(np.uint8)
            imageio.imwrite(osp.join(args.result_dir, f'lr_{i}.png'), lr)
            imageio.imwrite(osp.join(args.result_dir, f'hr_{i}.png'), hr)
            imageio.imwrite(osp.join(args.result_dir, f'sr_{i}.png'), sr)
    print('psnr: {:.2f}'.format(np.mean(psnrs)))


if __name__ == '__main__':
    main()
