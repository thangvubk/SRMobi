import os.path as osp
import time

import numpy as np
import tensorflow as tf
from config import get_args
from dataset import DataLoader, Dataset
from model import EDSR
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import compute_psnr, get_root_logger, update_tfboard


def main():
    args = get_args()

    writer = SummaryWriter(args.work_dir)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file)

    train_dataset = Dataset(
        dataset=args.train_dataset,
        split='train',
        crop_cfg=dict(type='random', patch_size=args.patch_size),
        flip_and_rotate=True)
    val_dataset = Dataset(
        dataset=args.valid_dataset, split='valid', override_length=args.num_valids, crop_cfg=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = EDSR(channels=args.num_channels)
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    best_psnr = 0

    for epoch in range(1, args.num_epochs + 1):
        losses = []
        for lr, hr in tqdm(train_loader):
            lr = tf.constant(lr, dtype=tf.float32)
            hr = tf.constant(hr, dtype=tf.float32)
            with tf.GradientTape() as tape:
                sr = model(lr)
                loss = loss_fn(hr, sr)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            losses.append(loss.numpy())
        logger.info(f'Epoch {epoch} - loss: {np.mean(losses)}')
        writer.add_scalar('loss', np.mean(losses), epoch)

        # eval
        if epoch % args.eval_freq == 0 or epoch == args.num_epochs:
            logger.info('Evaluating...')
            psnrs = []
            for i, (lr, hr) in enumerate(val_loader):
                lr = tf.constant(lr, dtype=tf.float32)
                hr = tf.constant(hr, dtype=tf.float32)
                sr = model(lr)
                cur_psnr = compute_psnr(sr, hr)
                psnrs.append(cur_psnr)
                update_tfboard(writer, i, lr, hr, sr, epoch)
            psnr = np.mean(psnrs)
            if psnr > best_psnr:
                best_psnr = psnr
            model.save_weights(osp.join(args.work_dir, f'epoch_{epoch}'))
            logger.info('psnr: {:.2f} (best={:.2f})'.format(psnr, best_psnr))
            writer.add_scalar('psnr', psnr, epoch)
            writer.flush()


if __name__ == '__main__':
    main()
