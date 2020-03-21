"""
train the dyolo model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes,  yolo_loss, gyolo_body
from yolo3.utils import get_random_data


def _main():
    annotation_path = 'dataset/train.txt'
    log_dir = 'logs/bgyolo000/'
    classes_path = 'model_data/drive_classes.txt'
    anchors_path = 'model_data/drive_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (384,384) # multiple of 128, hw

    model = create_model(input_shape, anchors, num_classes,
        freeze_body=1, weights_path='model_data/trained_weights_stage_1.h5') # make sure you know what you freeze
    # TensorBoard 可视化
    logging = TensorBoard(log_dir=log_dir)
    # ModelCheckpoint存储最优的模型
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    # ReduceLROnPlateau 当监视的loss不变时，学习率减少 factor：学习速率降低的因素。new_lr = lr * factor ; min_lr：学习率的下限。
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    # EarlyStopping 早停止;patience当连续多少个epochs时验证集精度不再变好终止训练，这里选择了10
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

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
        model.compile(optimizer=Adam(lr=1e-3),loss={'yolo_loss': lambda y_true, y_pred: y_pred})#模型输入即为loss
        # model.summary()
        # Total params: 51, 161, 738
        # Trainable params: 10, 534, 954
        # Non - trainable params: 40, 626, 784
        batch_size = 4
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'gyolo_trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if False:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')
        batch_size = 4 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'gyolo_trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=0,
            weights_path='model_data/gyolo_trained_weights_stage_1.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//128, w//128, num_anchors, num_classes+5))]

    model_body = gyolo_body(image_input, num_anchors, num_classes)
    print('Create gyolo model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body == 0:
            pass
        elif freeze_body in [1, 2]:
            # Freeze ghost body or freeze all but 3 output layers.
            num = (179, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [model_body.output, *y_true]) # 使用*号，可以将list解包
    print("model_loss create success")
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

def main():
    gyolo_model = gyolo_body(Input(shape=(384, 384, 3)), 3, 10)
    print("dyolo_model have",len(gyolo_model.layers),"layers")
    gyolo_model.summary()


# gyolo_model have 294 layers
# Total params: 12,634,661
# Trainable params: 12,611,077
# Non-trainable params: 23,584
if __name__ == '__main__':
    # main()
    _main()
