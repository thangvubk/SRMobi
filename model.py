import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer
from tensorflow.keras.models import Model, Sequential


class BasicBlock(Layer):  # TODO inherit layer

    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(channels, 3, padding='same', activation='relu')
        self.conv2 = Conv2D(channels, 3, padding='same')

    def call(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        return out


class Upsampler(Layer):

    def __init__(self, channels):
        super(Upsampler, self).__init__()
        self.conv1 = Conv2D(channels * 4, 3, padding='same')
        self.conv2 = Conv2D(channels * 4, 3, padding='same')

    def call(self, x):
        out = self.conv1(x)
        out = tf.nn.depth_to_space(out, 2)
        out = self.conv2(out)
        out = tf.nn.depth_to_space(out, 2)
        return out


class EDSR(Model):

    def __init__(self, num_blocks=16, channels=32):
        super(EDSR, self).__init__()
        self.conv1 = Conv2D(channels, 3, padding='same')
        body = [BasicBlock(channels) for _ in range(num_blocks)]
        self.body = Sequential(body)
        self.upsample = Upsampler(channels)
        self.conv_last = Conv2D(3, 3, padding='same')

    def call(self, x):
        out = self.conv1(x)
        out = self.body(out)
        out = self.upsample(out)
        out = self.conv_last(out)
        return out
