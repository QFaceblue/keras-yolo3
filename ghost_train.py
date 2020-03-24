"""
train the ghostnet for your own dataset.
"""
import keras
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from ghostnet import ghostnet
from yolo3.utils import get_random_data


def _main():
    annotation_path = 'dataset/train.txt'
    log_dir = 'logs/ghostnet/000/'
    classes_path = 'model_data/drive_classes.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    input_shape = (384,384) # multiple of 128, hw

    model = create_model(num_classes, load_pretrained=True,
        freeze_body=1, weights_path='logs/cifar10/555/ghostnet_cifar10.h5') # make sure you know what you freeze
    # TensorBoard 可视化
    logging = TensorBoard(log_dir=log_dir)
    # ModelCheckpoint存储最优的模型
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5',
        monitor='val_acc', save_weights_only=True, save_best_only=True, period=5)
    # ReduceLROnPlateau 当监视的loss不变时，学习率减少 factor：学习速率降低的因素。new_lr = lr * factor ; min_lr：学习率的下限。
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1)
    # EarlyStopping 早停止;patience当连续多少个epochs时验证集精度不再变好终止训练，这里选择了10。
    # 一般先经过一段时间训练后，在使用，应为训练前期loss不稳定，可能会在loss还没稳定时就停止了。
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        # for i in range(len(model.layers)):
        #     model.layers[i].trainable = True
        # use custom yolo_loss Lambda layer.
        # loss后类似’binary_crossentropy’、’mse’等代称
        # loss为函数名称的时候，不带括号
        # 函数参数必须为(y_true, y_pred, **kwards)的格式
        # 不能直接使用tf.nn.sigmoid_cross_entropy_with_logits等函数，
        # 因为其参数格式为(labels=None,logits=None)，需要指定labels =、logits = 这两个参数
        model.compile(optimizer=Adam(lr=1e-1),loss='categorical_crossentropy',metrics=['accuracy'])#模型输入即为loss
        # model.summary()

        batch_size = 8
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=32,
                initial_epoch=0,
                callbacks=[logging, checkpoint,reduce_lr, early_stopping])
        model.save_weights(log_dir + 'ghostnet_trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-2), loss='categorical_crossentropy',metrics=['accuracy']) # recompile to apply the change
        print('Unfreeze all of the layers.')
        batch_size = 8 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=96,
            initial_epoch=32,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'ghostnet_trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def create_model(num_classes, load_pretrained=True, freeze_body=1,
            weights_path='logs/cifar10/555/ghostnet_cifar10.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    model = ghostnet(mul=1, ratio=2, class_num=num_classes)
    print('Create ghostnet model with {} classes.'.format(num_classes))

    if load_pretrained:
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze ghost body or freeze all but 3 output layers.
            num = (179, len(model.layers)-3)[freeze_body-1]
            for i in range(num): model.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model.layers)))
    print("model_loss create success")
    return model


def data_generator(annotation_lines, batch_size, input_shape, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        label_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            label = keras.utils.to_categorical(box[0][-1], num_classes)
            label_data.append(label)
            i = (i+1) % n
        image_data = np.array(image_data)
        label_data = np.array(label_data)
        yield image_data, label_data

def data_generator_wrapper(annotation_lines, batch_size, input_shape, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, num_classes)

def main():
    pass
if __name__ == '__main__':
    _main()
