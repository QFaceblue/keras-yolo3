import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential,Model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,GlobalAveragePooling2D, Input, Activation,Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from ghostnet import  ghost_body,ghostBottleneck

# Total params: 2,756,866
# Trainable params: 2,736,866
# Non-trainable params: 20,000

# Total params: 2,534,466
# Trainable params: 2,517,026
# Non-trainable params: 17,440
def ghost_model():
    inputs = Input(shape=(32, 32, 3))
    x = ghost_body(inputs,mul=1,ratio=2)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1280)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(10)(x)
    x = Activation("softmax")(x)
    model = Model(inputs,x)
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    # 评价函数用于评估当前训练模型的性能。当模型编译后（compile），评价函数应该作为 metrics 的参数来输入。
    # 评价函数和 损失函数 相似，只不过评价函数的结果不会用于训练过程中。
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# Total params: 72,250
# Trainable params: 69,786
# Non-trainable params: 2,464
def cifar10_model():
    cfgs = [
        # k, t, c, SE, s
        [3, 16, 16, 0, 1],
        [3, 48, 24, 0, 2],
        [3, 72, 24, 0, 1],
        [5, 72, 40, 1, 2],
        [5, 120, 40, 1, 1],
        [3, 240, 80, 0, 2],
        [3, 200, 80, 0, 1]
    ]
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(16 , 3, strides=2, padding="SAME", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # cifar cfgs
    for k, exp_size, c, use_se, s in cfgs:
        output_channel = c
        hidden_channel = exp_size
        x = ghostBottleneck(x, hidden_channel, output_channel, kernel_size=k, use_se=use_se, strides=s)
    x = GlobalAveragePooling2D()(x)
    # x = Dense(128)(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    # x = Dropout(0.2)(x)
    x = Dense(10)(x)
    x = Activation("softmax")(x)
    model = Model(inputs,x)
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    # 评价函数用于评估当前训练模型的性能。当模型编译后（compile），评价函数应该作为 metrics 的参数来输入。
    # 评价函数和 损失函数 相似，只不过评价函数的结果不会用于训练过程中。
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# Total params: 136,826
# Trainable params: 132,986
# Non-trainable params: 3,840
def cifar10_model2():
    cfgs = [
        # k, t, c, SE, s
        [3, 16, 16, 0, 1],
        [3, 48, 24, 0, 2],
        [3, 72, 24, 0, 1],
        [5, 72, 40, 1, 2],
        [3, 120, 40, 1, 1],
        [3, 240, 80, 0, 2],
        [3, 240, 80, 0, 1],
        [3, 320, 160, 0, 2]
    ]
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(8 , 3, strides=1, padding="SAME", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # cifar cfgs
    for k, exp_size, c, use_se, s in cfgs:
        output_channel = c
        hidden_channel = exp_size
        x = ghostBottleneck(x, hidden_channel, output_channel, kernel_size=k, use_se=use_se, strides=s)
    x = GlobalAveragePooling2D()(x)
    # x = Dense(128)(x)
    # x = BatchNormalization()(x)
    # x = Activation("relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(10)(x)
    x = Activation("softmax")(x)
    model = Model(inputs,x)
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    # 评价函数用于评估当前训练模型的性能。当模型编译后（compile），评价函数应该作为 metrics 的参数来输入。
    # 评价函数和 损失函数 相似，只不过评价函数的结果不会用于训练过程中。
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test

def scheduler(epoch):
    if epoch < 30:
        return 0.1
    if epoch < 60:
        return 0.01
    if epoch < 90:
        return 0.005
    return 0.001

def scheduler2(epoch):
    if epoch < 81:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.001

def scheduler3(epoch):
    if epoch < 50:
        return 0.1
    if epoch < 90:
        return 0.01
    return 0.001

# without preprocess
def main():
    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    log_dir = 'logs/cifar10/333/'
    # build network
    # model = ghost_model()
    model = cifar10_model2()
    # print(model.summary())
    # print(model.output)
    # set callback
    checkpoint = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5',
        monitor='val_acc', save_weights_only=True, save_best_only=True, period=10)
    tb_cb = TensorBoard(log_dir=log_dir)
    change_lr = LearningRateScheduler(scheduler)  # 调整lr
    # 根据条件提前停止
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1)
    cbks = [checkpoint, tb_cb, change_lr]

    # start train
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=80,
              callbacks=cbks,
              validation_data=(x_test, y_test),
              shuffle=True)

    # save model
    model.save(log_dir + 'ghostnet_cifar10.h5')

# with preprocess
def main2():
    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)

    log_dir = 'logs/cifar10/666/'
    # build network
    model = ghost_model()
    # print(model.summary())
    # print(model.output)
    # set callback
    checkpoint = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5',
        monitor='val_acc', save_weights_only=True, save_best_only=True, period=10)
    tb_cb = TensorBoard(log_dir=log_dir)
    change_lr = LearningRateScheduler(scheduler2)  # 调整lr
    cbks = [checkpoint, tb_cb, change_lr]
    # set data augmentation
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)

    datagen.fit(x_train)

    # start training
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                         steps_per_epoch=len(x_train)//128,
                         epochs=110,
                         callbacks=cbks,
                         validation_data=(x_test, y_test))
    # save model
    model.save(log_dir + 'ghostnet_cifar10.h5')
if __name__ == '__main__':
    main2()