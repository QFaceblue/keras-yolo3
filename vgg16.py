import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, DepthwiseConv2D, Concatenate
from keras.initializers import he_normal
from keras import optimizers, Input
from keras.callbacks import LearningRateScheduler, TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file

num_classes  = 10
batch_size   = 128
epochs       = 200
iterations   = 50000//batch_size
dropout      = 0.5
weight_decay = 0.0001
class_num    = 10
log_filepath = 'logs/cifar10/ghost_vgg16/1024/'

from keras import backend as K
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

# ghost模块
def GhostModule(x, filters, kernel_size=1, dw_size=3, ratio=2, padding='SAME', strides=1, use_bias=False, relu=False,
                bn=False,
                kernel_initializer='he_normal', kernel_regularizer=None):
    assert ratio >= 1
    init_channels = np.math.ceil(filters / ratio)
    base = Conv2D(init_channels, kernel_size, strides=strides, padding=padding,
                  kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, use_bias=use_bias)(x)
    if bn:
        base = BatchNormalization()(base)
    if relu:
        base = Activation("relu")(base)
    if ratio == 1:
        return base
    ghost = DepthwiseConv2D(dw_size, strides=1, padding=padding, depth_multiplier=(ratio - 1),
                            depthwise_initializer=kernel_initializer)(base)
    if bn:
        ghost = BatchNormalization()(ghost)
    if relu:
        ghost = Activation("relu")(ghost)
    # base_ghost = K.concatenate([base, ghost], 3)
    base_ghost = Concatenate(3)([base, ghost])  # 使用keras layer包装
    return base_ghost

def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 160:
        return 0.01
    return 0.001
# download pre-trianing weights
# WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
# filepath = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')

# data loading
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# data preprocessing
x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)

# build model
def vgg16(class_num =10):
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv1', input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # model modification for cifar-10
    model.add(Flatten(name='flatten'))
    model.add(Dense(1024, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc_cifa10'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1024, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(class_num, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_cifa10'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model

def ghost_vgg16(class_num =10):
    model = Sequential()
    input = Input(shape=(32, 32, 3))
    # Block 1
    x = GhostModule(input, 64, 3, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal())
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GhostModule(x, 64, 3, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal())
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = GhostModule(x, 128, 3, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal())
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GhostModule(x, 128, 3, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal())
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = GhostModule(x, 256, 3, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal())
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GhostModule(x, 256, 3, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal())
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GhostModule(x,256, 1, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal())
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = GhostModule(x, 512, 3, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal())
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GhostModule(x, 512, 3, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal())
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GhostModule(x, 512, 1, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal())
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = GhostModule(x, 512, 3, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal())
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GhostModule(x,512, 3, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal())
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GhostModule(x, 512, 1, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal())
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # model modification for cifar-10
    x = Flatten()(x)
    x = Dense(1024, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(1024, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(class_num, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_cifa10')(x)
    x = BatchNormalization()(x)
    x = Activation('softmax')(x)
    model = Model(input,x)
    return model
# -------- optimizer setting -------- #
def main():
    # model = vgg16(class_num)
    model = ghost_vgg16(class_num)
    # load pretrained weight from VGG16 by name
    # model.load_weights(filepath, by_name=True)
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    # change_lr = LearningRateScheduler(scheduler)
    # cbks = [change_lr, tb_cb]
    checkpoint = ModelCheckpoint(
        log_filepath + 'vgg19-ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5',
        monitor='val_acc', save_weights_only=True, save_best_only=True, period=20)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1)  # min_lr=1e-4,
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    cbks = [checkpoint, tb_cb, reduce_lr, early_stopping]
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_test, y_test))
    model.save(log_filepath+'ghost_vgg16.h5')
    # model.save('ghost_vgg16.h5')

# vgg16 fc1:4096;fc2:4096
# Total params: 28,969,330
# Trainable params: 28,944,478
# Non-trainable params: 24,852

# vgg16 fc1:1024;fc2:1024
# Total params: 11,606,386
# Trainable params: 11,593,822
# Non-trainable params: 12,564

# ghost_vgg16 fc1:4096;fc2:4096
# Total params: 23,990,290
# Trainable params: 23,965,438
# Non-trainable params: 24,852

# ghost_vgg16 fc1:1024;fc2:1024
# Total params: 6,627,346
# Trainable params: 6,614,782
# Non-trainable params: 12,564
def _main():
    model =vgg16(class_num)
    # model = ghost_vgg16(class_num)
    model.summary()

if __name__ == '__main__':
    main()
    # _main()