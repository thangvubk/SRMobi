import errno
import logging
import math
import os

import numpy as np
import tensorflow as tf


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger('SRMobi')
    if logger.hasHandlers():
        return logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    return logger


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def rgb2y(rgb):
    coeff = [65.738 / 256, 129.057 / 256, 25.064 / 256]
    offset = 16
    coeff = tf.constant(coeff)
    y = tf.reduce_sum(rgb * coeff, axis=-1) + offset
    return y


def compute_psnr(out, lbl):
    out = tf.round(tf.clip_by_value(out, 0, 255))
    lbl = tf.round(tf.clip_by_value(lbl, 0, 255))
    out = rgb2y(out)
    lbl = rgb2y(lbl)
    out = tf.round(tf.clip_by_value(out, 0, 255))
    lbl = tf.round(tf.clip_by_value(lbl, 0, 255))
    diff = out - lbl
    rmse = tf.math.sqrt(tf.reduce_mean(diff**2)).numpy()
    psnr = 20 * math.log10(255 / rmse)
    return psnr


def update_tfboard(writer, img_id, lr, hr, sr, epoch):
    # image shape (1, H, W, C)
    lr = lr.numpy()[0].clip(0, 255).astype(np.uint8).transpose(2, 0, 1)
    hr = hr.numpy()[0].clip(0, 255).astype(np.uint8).transpose(2, 0, 1)
    sr = sr.numpy()[0].clip(0, 255).astype(np.uint8).transpose(2, 0, 1)
    writer.add_image(f'LR_{img_id}', lr, epoch)
    writer.add_image(f'HR_{img_id}', hr, epoch)
    writer.add_image(f'SR_{img_id}', sr, epoch)
