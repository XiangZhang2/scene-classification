# coding=utf-8
"""DarkNet53 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, Concatenate, MaxPooling2D, Input, GlobalAveragePooling2D, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from utils import compose

@wraps(Conv2D) #本来下面函数的名字变成了DraknetConv2D，使用了wraps以后，name还是Conv2D
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)} #给每一层添加了L2正则化
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs) #是更新kwargs的意思么？
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    """每一个卷积层都是由BN和LeakyReLU组成的，BN一般在激活函数之前设置，没有偏置项，why？"""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    #一方面是为了用left and top padding，另一方面因为下面在进行3x3卷积的时候padding=same
    #其实还是为了使用这种填充方法吧，好处是？
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x


def darknet_body(inputs):
    '''Darknent body having 52 Convolution2D layers'''
    '''从输入到输出的一个完整网络'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(inputs) #input(416,416,3),output(416,416,32)
    x = resblock_body(x, 64, 1)  #output(208,208,64)
    x = resblock_body(x, 128, 2)  #output(104,104,128)
    x = resblock_body(x, 256, 8)  #output(52,52,256)
    x = resblock_body(x, 512, 8)  #output(26,26,512)
    y = resblock_body(x, 1024, 4)  #output(13,13,1024)
    model = Model(inputs, y)
    return model


def darknet(darknet_body, classes):
    
    # darknet = Model(inouts, darknet_body(inputs))
    x = GlobalAveragePooling2D(name='GAP')(darknet_body.output) #1024
    x = Dense(classes, activation='softmax', name='fc6')(x)
    #x = Activation('softmax', name='prob')(x)
    model = Model(darknet_body.input, x, name='darknet')

    return model































