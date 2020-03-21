"""ghostnet Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
import math
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, GlobalAveragePooling2D, \
    Activation, SeparableConv2D, DepthwiseConv2D,Input,Add,Lambda,Reshape,Multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from yolo3.utils import compose

def clip(x):
    return tf.clip_by_value(x,0, 1)
def clip2(x,min,max):
    return tf.clip_by_value(x,min,max)
# Lambda层
# keras.layers.core.Lambda(function, output_shape=None, mask=None, arguments=None)
# 本函数用以对上一层的输出施以任何Theano/TensorFlow表达式
# 参数
# function：要实现的函数，该函数仅接受一个变量，即上一层的输出
# output_shape：函数应该返回的值的shape，可以是一个tuple，也可以是一个根据输入shape计算输出shape的函数
# mask: 掩膜
# arguments：可选，字典，用来记录向函数中传递的其他关键字参数

def _make_divisible(v, divisor=4, min_value=None):
    """
    It ensures that all layers have a channel number that is divisible by 4
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
# 注意力机制
def SELayer(x,out_dim, ratio=4):
    # print("use seLayer,输入形状为",x.shape)
    #b, w, h, c = x.shape
    # print(b, w, h, c)
    squeeze = GlobalAveragePooling2D()(x)
    squeeze =Reshape((1,1,out_dim))(squeeze)
    # squeeze = K.reshape(squeeze, (-1, 1, 1, out_dim))
    excitation = Conv2D(int(out_dim/ ratio), 1)(squeeze)
    excitation = Activation("relu")(excitation)
    excitation = Conv2D(out_dim, 1)(excitation)
    # excitation = tf.clip_by_value(excitation, 0, 1)
    # excitation = Lambda(clip)(excitation)
    excitation = Lambda(clip2,arguments={"min":0,"max":1})(excitation)
    # scale = x * excitation
    scale = Multiply()([x,excitation])
    return scale


# ghost模块
def GhostModule(x, filters, kernel_size=1, dw_size=3, ratio=2, padding='SAME', strides=1, use_bias=False, relu=True,bn=True,
                kernel_initializer='he_normal',kernel_regularizer=None):
    assert ratio>=1
    init_channels = math.ceil(filters / ratio)
    base = Conv2D(init_channels, kernel_size, strides=strides, padding=padding,
                  kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,use_bias=use_bias)(x)
    if bn:
        base = BatchNormalization()(base)
    if relu:
        base = Activation("relu")(base)
    if ratio == 1:
        return base
    ghost = DepthwiseConv2D(dw_size, strides=1, padding=padding, depth_multiplier=(ratio-1),
                            depthwise_initializer=kernel_initializer)(base)
    if bn:
        ghost = BatchNormalization()(ghost)
    if relu:
        ghost = Activation("relu")(ghost)
    # base_ghost = K.concatenate([base, ghost], 3)
    base_ghost = Concatenate(3)([base, ghost]) #使用keras layer包装
    return base_ghost

def ghostBottleneck(x, hidden_dim, out_dim, kernel_size=3, ratio=2,strides=1, use_se=False):
    assert strides in [1, 2]
    input_dim = int(x.shape[3])
    hidden = GhostModule(x,hidden_dim,kernel_size=1, ratio=ratio)
    if strides ==2:
        hidden = DepthwiseConv2D(kernel_size, strides=2, padding='SAME')(hidden)
    if use_se:
        # pass
        hidden = SELayer(hidden,hidden_dim)
    res = GhostModule(hidden,out_dim,kernel_size=1, ratio=ratio,relu=False)
    shortcut = x
    if strides ==2:
        shortcut = DepthwiseConv2D(3, strides=2, padding='SAME')(shortcut)
    if input_dim != out_dim:
        shortcut = Conv2D(out_dim, 1)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    out = Add()([res,shortcut]) #使用keras layer包装
    # out = res+shortcut
    return out

def my_ghostBottleneck(x, hidden_dim, out_dim, kernel_size=3, ratio=2,strides=1, kernel_regularizer=None,use_se=False):
    assert strides in [1, 2]
    input_dim = int(x.shape[3])
    hidden = GhostModule(x,hidden_dim,kernel_size=1, kernel_regularizer=kernel_regularizer,ratio=ratio)
    if strides ==2:
        hidden = DepthwiseConv2D(kernel_size, strides=2, depthwise_regularizer=kernel_regularizer,padding='SAME')(hidden)
    if use_se:
        # pass
        hidden = SELayer(hidden,hidden_dim)
    res = GhostModule(hidden,out_dim,kernel_size=kernel_size if strides==1 else 1,kernel_regularizer=kernel_regularizer, ratio=ratio,relu=False)
    shortcut = x
    if strides ==2:
        shortcut = DepthwiseConv2D(3, strides=2, padding='SAME')(shortcut)
    if input_dim != out_dim:
        shortcut = Conv2D(out_dim, 1)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    out = Add()([res,shortcut]) #使用keras layer包装
    # out = res+shortcut
    return out

def ghost_body(x,mul=1,ratio=2):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, s
        [3,  16,  16, 0, 1],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 2],
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]
    # input
    # inputs = Input(shape=(384,384,3,))
    # make first layer
    print("input shape:",x.shape)
    x = Conv2D(_make_divisible(16*mul), 3, strides=2, padding="SAME",use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # print("first layer output shape:", x.shape)
    # ghostnet cfgs
    for k, exp_size, c, use_se, s in cfgs:
        output_channel = _make_divisible(c * mul)
        hidden_channel = _make_divisible(exp_size * mul)
        x = ghostBottleneck(x,hidden_channel,output_channel,kernel_size=k,use_se=use_se,strides=s,ratio=ratio)
    # add [5, 960, 160, 0, 2]
    # x = ghostBottleneck(x,960,160,kernel_size=5,use_se=0,strides=2)
    return x
# ghost_body have 279 layers
# Total params: 2,535,032
# Trainable params: 2,517,592
# Non-trainable params: 17,440

# set kernel_size = kernel_size in the second ghost module in ghostbottleneck
# Total params: 12,075,256
# Trainable params: 12,057,816
# Non-trainable params: 17,440
if __name__ == '__main__':
    # x = K.zeros((5, 4, 3, 2))
    # y = SELayer(x,2)
    # z = GhostModule(x,10)
    # print(y.shape)
    # print(z.shape)
    # x = K.zeros((5, 10, 10, 5))
    # y = GhostBottleneck(x,10,6,use_se=True)
    # z = GhostBottleneck(x,10,6,strides=2)
    # print(y.shape)
    # print(z.shape)

    inputs = Input(shape=(384, 384, 3))
    model = Model(inputs=inputs,outputs=ghost_body(inputs))
    model.summary()
    print("ghost_body have", len(model.layers), "layers")

    # # model = Model(inputs=inputs, outputs=GhostModule(inputs,10))
    # # model = Model(inputs=inputs, outputs=GhostBottleneck(inputs,960,160,kernel_size=5,use_se=0,strides=2))
    # # model = Model(inputs=inputs, outputs=SELayer(inputs, 3))


    # x = K.zeros((5, 384, 384, 3))
    # y = ghost_body(x)
    # print(y.shape,y.name,y.dtype)