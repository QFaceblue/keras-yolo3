import os

import tensorflow as tf
import pprint # 使用pprint 提高打印的可读性

from keras.legacy import layers
from keras.utils import Progbar
import h5py

def print_tensorflow_wegiths(weight_file_path):
    NewCheck =tf.train.NewCheckpointReader(weight_file_path)# "logs/ghostnet/ghostnet_checkpoint"
    # # 打印模型中所有变量
    # print("debug_string:\n")
    # pprint.pprint(NewCheck.debug_string().decode("utf-8"))# 输出中3个字段分别为：变量名，类型和形状
    # # 获取变量中的值--get_tensor
    # print("get_tensor:\n")
    # pprint.pprint(NewCheck.get_tensor("MobileNetV2/Conv2d_0/W"))
    # 打印所有变量类型 输出：变量名，类型
    # print("get_variable_to_dtype_map\n")
    # pprint.pprint(NewCheck.get_variable_to_dtype_map())
    # 打印所有变量形状 输出：变量名，形状
    print("get_variable_to_shape_map\n")
    pprint.pprint(NewCheck.get_variable_to_shape_map())



def print_keras_wegiths(weight_file_path):
    f = h5py.File(weight_file_path)  # 读取weights h5文件返回File类
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))  # 输出储存在File类中的attrs信息，一般是各层的名称

        for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items(): # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
                print("      {}: {}".format(key, value))

            # print("    Dataset:")
            # for name, d in g.items(): # 读取各层储存具体信息的Dataset类
            #     print("      {}: {}".format(name, d.value.shape)) # 输出储存在Dataset中的层名称和权重，也可以打印dataset的attrs，但是keras中是空的
            #     print("      {}: {}".format(name. d.value))
    finally:
        f.close()

# def load_weights_from_tf_checkpoint(model, checkpoint_file, background_label):
#     print('Load weights from tensorflow checkpoint')
#     progbar = Progbar(target=len(model.layers))
#
#     reader = tf.train.NewCheckpointReader(checkpoint_file)
#     for index, layer in enumerate(model.layers):
#         progbar.update(current=index)
#
#         if isinstance(layer, layers.convolutional.SeparableConv2D):
#             depthwise = reader.get_tensor('{}/depthwise_weights'.format(layer.name))
#             pointwise = reader.get_tensor('{}/pointwise_weights'.format(layer.name))
#
#             if K.image_data_format() == 'channels_first':
#                 depthwise = convert_kernel(depthwise)
#                 pointwise = convert_kernel(pointwise)
#
#             layer.set_weights([depthwise, pointwise])
#         elif isinstance(layer, layers.convolutional.Convolution2D):
#             weights = reader.get_tensor('{}/weights'.format(layer.name))
#
#             if K.image_data_format() == 'channels_first':
#                 weights = convert_kernel(weights)
#
#             layer.set_weights([weights])
#         elif isinstance(layer, layers.BatchNormalization):
#             beta = reader.get_tensor('{}/beta'.format(layer.name))
#             gamma = reader.get_tensor('{}/gamma'.format(layer.name))
#             moving_mean = reader.get_tensor('{}/moving_mean'.format(layer.name))
#             moving_variance = reader.get_tensor('{}/moving_variance'.format(layer.name))
#
#             layer.set_weights([gamma, beta, moving_mean, moving_variance])
#         elif isinstance(layer, layers.Dense):
#             weights = reader.get_tensor('{}/weights'.format(layer.name))
#             biases = reader.get_tensor('{}/biases'.format(layer.name))
#
#             if background_label:
#                 layer.set_weights([weights, biases])
#             else:
#                 layer.set_weights([weights[:, 1:], biases[1:]])
#
#
# def load_pretrained_weights(model, fname, origin, md5_hash, background_label=False, cache_dir=None):
#     """Download and convert tensorflow checkpoints"""
#     # origin = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz'
#
#     if cache_dir is None:
#         cache_dir = os.path.expanduser(os.path.join('~', '.keras', 'models'))
#
#     weight_path = os.path.join(cache_dir, '{}_{}_{}.h5'.format(model.name, md5_hash, K.image_data_format()))
#
#     if os.path.exists(weight_path):
#         model.load_weights(weight_path)
#     else:
#         if not os.path.exists(cache_dir):
#             os.makedirs(cache_dir)
#             print(cache_dir)
#         path = get_file(fname, origin=origin, extract=True, md5_hash=md5_hash, cache_dir=cache_dir, cache_subdir='.')
#         checkpoint_file = os.path.join(path, '..', 'model.ckpt')
#         load_weights_from_tf_checkpoint(model, checkpoint_file, background_label)
#         model.save_weights(weight_path)

def main():
    tf_w_path = "logs/ghostnet/ghostnet_checkpoint"
    keras_w_path = "logs/cifar10/555/ghostnet_cifar10.h5"
    # print_tensorflow_wegiths(tf_w_path)
    print_keras_wegiths(keras_w_path)

if __name__ == '__main__':
    main()