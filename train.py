import tensorflow as tf
from config import get_args
from dataset import DataLoader, Dataset
from model import EDSR
from utils import get_root_logger


def main():
    logger = get_root_logger()
    args = get_args()
    train_dataset = Dataset()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = EDSR(channels=32)
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    for lr, hr in train_loader:
        lr = tf.constant(lr, dtype=tf.float32)
        hr = tf.constant(hr, dtype=tf.float32)
        with tf.GradientTape() as tape:
            sr = model(lr)
            loss = loss_fn(hr, sr)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        logger.info(f'loss: {loss.numpy()}')


if __name__ == '__main__':
    main()
