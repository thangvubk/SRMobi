import tensorflow as tf


def basic_block(x, channels):
    layer = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(channels, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(channels, 3, padding='same')
    ])

    out = layer(x)
    return out + x


def upsampler(x, channels):
    conv1 = tf.keras.layers.Conv2D(channels * 4, 3, padding='same')
    conv2 = tf.keras.layers.Conv2D(channels * 4, 3, padding='same')
    conv3 = tf.keras.layers.Conv2D(3, 3, padding='same')

    out = conv1(x)
    out = tf.nn.depth_to_space(out, 2)
    out = conv2(out)
    out = tf.nn.depth_to_space(out, 2)
    out = conv3(out)
    return out


def resnet(x, blocks=16, channels=32):
    conv1 = tf.keras.layers.Conv2D(channels, 3, padding='same', activation='relu')

    out = conv1(x)
    for _ in range(blocks):
        out = basic_block(out, channels)
    out = upsampler(out, channels)
    return out


if __name__ == '__main__':
    from dataset import Dataset, DataLoader
    dataloader = DataLoader(Dataset(), 2)
    for lr, hr in dataloader:
        import pdb
        pdb.set_trace()
        lr = tf.constant(lr, dtype=tf.float32)
        hr = tf.constant(hr, dtype=tf.float32)
        sr = resnet(lr, 16, 32)
