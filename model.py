import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D


class BasicBlock(Model):

    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(channels, 3, padding='same', activation='relu')
        self.conv2 = Conv2D(channels, 3, padding='same')

    def call(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        return out


class Upsampler(Model):

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

    def __init__(self, channels):
        super(EDSR, self).__init__()
        self.conv1 = Conv2D(channels, 3, padding='same')
        self.basic_block_1 = BasicBlock(channels)
        self.basic_block_2 = BasicBlock(channels)
        self.basic_block_3 = BasicBlock(channels)
        self.basic_block_4 = BasicBlock(channels)
        self.basic_block_5 = BasicBlock(channels)
        self.basic_block_6 = BasicBlock(channels)
        self.basic_block_7 = BasicBlock(channels)
        self.basic_block_8 = BasicBlock(channels)
        self.basic_block_9 = BasicBlock(channels)
        self.basic_block_10 = BasicBlock(channels)
        self.basic_block_11 = BasicBlock(channels)
        self.basic_block_12 = BasicBlock(channels)
        self.basic_block_13 = BasicBlock(channels)
        self.basic_block_14 = BasicBlock(channels)
        self.basic_block_15 = BasicBlock(channels)
        self.basic_block_16 = BasicBlock(channels)
        self.upsample = Upsampler(channels)
        self.conv_last = Conv2D(3, 3, padding='same')

    def call(self, x):
        out = self.conv1(x)
        out = self.basic_block_1(out)
        out = self.basic_block_2(out)
        out = self.basic_block_3(out)
        out = self.basic_block_4(out)
        out = self.basic_block_5(out)
        out = self.basic_block_6(out)
        out = self.basic_block_7(out)
        out = self.basic_block_8(out)
        out = self.basic_block_9(out)
        out = self.basic_block_10(out)
        out = self.basic_block_11(out)
        out = self.basic_block_12(out)
        out = self.basic_block_13(out)
        out = self.basic_block_14(out)
        out = self.basic_block_15(out)
        out = self.basic_block_16(out)
        out = self.upsample(out)
        out = self.conv_last(out)
        return out
