import multiprocessing
import os.path as osp

import tensorflow as tf
from config import get_args
from model import EDSR


def main():
    args = get_args()

    def _save_model():
        model = EDSR(num_blocks=args.num_blocks, channels=args.num_channels)
        assert args.checkpoint != '', 'checkpoint need to be specified'
        model.load_weights(args.checkpoint)

        inputs = tf.zeros(shape=(1, 256, 256, 3))
        model(inputs)
        model.save(args.result_dir)

    process = multiprocessing.Process(target=_save_model)
    process.start()
    process.join()
    converter = tf.lite.TFLiteConverter.from_saved_model(args.result_dir)
    tflite_model = converter.convert()
    with open(osp.join(args.result_dir, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)
    print('Model converted!')


if __name__ == '__main__':
    main()
