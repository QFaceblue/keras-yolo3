"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose
from ghostnet import ghost_body


@wraps(Conv2D)  # 将Conv2D的信息赋值给DarknetConv2D
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'  # python 赋值语句可以加条件，但不要使用":"
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks): # total_number_of_conv = 1 + num_blocks * 2
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x) #由于步长为2，上左各部1即可使featuremap减半，代替maxpooling
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x) # featuremap减半
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y

# 驾驶行为检测模型
# 在darknet基础上添加输出处理层，使featuremap缩小为darknet输出的1/4,总共缩小128
# 模型输入 batch_size * 384 * 384 * 3
# 模型输出 barch_size * 3 * 3 * num_anchors * (num_classes + 5)
def dyolo_body(inputs, num_anchors, num_classes):
    darknet = Model(inputs, darknet_body(inputs))
    x = DarknetConv2D_BN_Leaky(512, (1, 1))(darknet.output)
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # 由于步长为2，上左各部1即可使featuremap减半，代替maxpooling
    x = DarknetConv2D_BN_Leaky(1024, (3, 3), strides=(2, 2))(x)  # featuremap减半
    x = DarknetConv2D_BN_Leaky(512, (1, 1))(x)
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # 由于步长为2，上左各部1即可使featuremap减半，代替maxpooling
    # x = DarknetConv2D_BN_Leaky(1024, (3, 3), strides=(2, 2))(x)  # featuremap减半
    # y = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(1024, (3, 3), strides=(2, 2)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))(x)
    return Model(inputs, y)

# ghostnet驾驶行为检测模型
# 在ghostnet基础上添加输出处理层，使featuremap缩小为darknet输出的1/4,总共缩小128
# 模型输入 batch_size * 384 * 384 * 3
# 模型输出 barch_size * 3 * 3 * num_anchors * (num_classes + 5)
def gyolo_body(inputs, num_anchors, num_classes):
    ghostnet = Model(inputs, ghost_body(inputs))
    x = DarknetConv2D_BN_Leaky(512, (1, 1))(ghostnet.output)
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # 由于步长为2，上左各部1即可使featuremap减半，代替maxpooling
    x = DarknetConv2D_BN_Leaky(1024, (3, 3), strides=(2, 2))(x)  # featuremap减半
    x = DarknetConv2D_BN_Leaky(512, (1, 1))(x)
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # 由于步长为2，上左各部1即可使featuremap减半，代替maxpooling
    # x = DarknetConv2D_BN_Leaky(1024, (3, 3), strides=(2, 2))(x)  # featuremap减半
    # y = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(1024, (3, 3), strides=(2, 2)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))(x)
    return Model(inputs, y)

def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])


def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)
    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)
    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))([x2, x1])

    return Model(inputs, [y1, y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # 获得x，y的网格
    # (13, 13, 1, 2)
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1]) #tile不能改变原变量维度
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y]) #最后一维拼接
    grid = K.cast(grid, K.dtype(feats))

    # batch, height, width, num_anchors, num_classes + 5 （x_offset、y_offset、h和w、置信度、分类结果）
    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    # 偏移量加上所属网格，然后再除以整个网格大小得到相对于整个网格偏移量
    # 广播：在两个数组上运行时，NumPy会逐元素地比较它们的形状。它从尾随尺寸开始，并向前发展。
    # 两个尺寸兼容时： 他们是相等的，或者其中一个是1
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    # 大小偏移量乘以锚点框大小，再除以输入大小，得到相对输入大小的框大小
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1] # 对最后一维进行操作
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape)) # round()四舍五入；min()求张量中最小值；input_shape>=new_shape
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

# yolo_boxes_and_scores(yolo_outputs[0],anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
# feats.shape=(N,13,13,3*(num_classes+5))
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4]) #这里好像默认batch为1了，所以评估时只能输入一张图片。
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores # box_number* 4,box_number*num_classes


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    if num_layers ==1:
        anchor_mask = [[0, 1, 2]]
        input_shape = K.shape(yolo_outputs[0])[1:3] * 128
    else:
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
        input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    # 对每个特征层进行处理
    if num_layers ==1:
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs,
                                                    anchors[anchor_mask[0]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    else:
        for l in range(num_layers):
            _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                        anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
    # 将每个特征层的结果进行堆叠
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes): #分类别挑选框
        # TODO: use keras backend instead of tf.
        # 取出所有box_scores >= score_threshold的框，和成绩
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        # 非极大抑制，按照scores降序去掉box重合程度高的，由于scores与类别有关所以分类挑选
        # 按照score降序排列，去除和已选择框iou过高的
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        # 获取非极大抑制后的结果
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

def dyolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = 1
    if num_layers ==1:
        anchor_mask = [[0, 1, 2]]
        input_shape = K.shape(yolo_outputs)[1:3] * 128
    else:
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
        input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    # 对每个特征层进行处理
    if num_layers ==1:
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs,
                                                    anchors[anchor_mask[0]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    else:
        for l in range(num_layers):
            _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                        anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
    # 将每个特征层的结果进行堆叠
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes): #分类别挑选框
        # TODO: use keras backend instead of tf.
        # 取出所有box_scores >= score_threshold的框，和成绩
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        # 非极大抑制，按照scores降序去掉box重合程度高的，由于scores与类别有关所以分类挑选
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        # 获取非极大抑制后的结果
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

# return y_ture (n,13,13,3,5+class_num)
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    # all() 函数用于判断给定的可迭代参数 iterable 中的所有元素是否都为 TRUE，如果是返回 True，否则返回 False。
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    if num_layers == 1:
        anchor_mask = [[0,1,2]]
    else:
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    if num_layers == 1:
        grid_shapes = [input_shape // 128]
        y_true = [np.zeros((m, grid_shapes[0][0], grid_shapes[0][1], len(anchor_mask[0]), 5 + num_classes),
                       dtype='float32')]
    else:
        # 得到网格的shape为13,13;26,26;52,52
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        # y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    # [1,9,2]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # 对每一张图进行处理
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        # [n,1,2]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes
        # 计算真实框和哪个先验框最契合
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        # (n,9,1)
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor
    #输入：
    #args：是一个list 组合，包括了预测值和真实值，具体如下：
    #     args[:num_layers]--预测值yolo_outputs，
    #     args[num_layers:]--真实值y_true，
    #     yolo_outputs，y_true同样是list，分别是[y1,y2,y3]三个feature map 上的的预测结果,
    #     每个y都是m*grid*grid*num_anchors*(num_classes+5),作者原文是三层，分别是(13,13,3,25)\
    #     (26,26,3,25),(52,52,3,25)
    #anchors:输入预先选择好的anchors box，原文是9个box,三层feature map 各三个。
    #num_classes：原文分了20类
    #ignore_thresh=.5:如果一个default box与true box的IOU 小于ignore_thresh，
    #                 则作为负样本confidence 损失。
    #print_loss：loss的打印开关。
    #输出：一维张量。

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    if num_layers == 1:
        anchor_mask = [[0, 1, 2]]
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 128, K.dtype(y_true[0]))
    else:
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5] #切片不降维
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        # keras.backend.switch(condition, then_expression, else_expression) 根据一个标量值在两个操作之间切换。
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        # 预测框越大loss越小
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        # TensorArray可以看做是具有动态size功能的Tensor数组。通常都是跟while_loop或map_fn结合使用。
        # tf.TensorArray--相当于建立一个动态数组，size=1,是二维动态数组。
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            # tensor是N维度的tensor，mask是K维度的，注意K小于等于N，name可选项也就是这个操作的名字，
            # axis是一个0维度的int型tensor，表示的是从参数tensor的哪个axis开始mask，默认的情况下，
            # axis=0表示从第一维度进行mask，因此K+axis小于等于N。
            # return (N-K+1)-dimensional
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])  # b是第几张图，将置信率为0的其他参数清
            iou = box_iou(pred_box[b], true_box)  # 单张图片单个尺度算iou，即该层所有预测窗口
            # pred_box(13,13,3,4)与真实窗口true_box(设有j个)之间的IOU，输出为iou(13,13,3,j)
            best_iou = K.max(iou, axis=-1)  # 先取每个grid上多个anchor box上的最大的iou
            # best_iou(13,13,3)值是最大的iou
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(
                true_box)))  # 删掉小于阈值的BBOX,ignore_mask[b]存放的是pred_box(13,13,3,4)iou小于
            # ignore_thresh的grid，即ignore_mask[b]=[13,13,3],如果小于ignore_thresh，其值为0，大于为1

            return b + 1, ignore_mask

        # 判断预测框的iou小于ignore_thresh则认为该预测框没有与之对应的真实框
        # 则被认为是这幅图的负样本
        # while_loop(cond,body,loop_vars,...) Repeat `body` while the condition `cond` is true.
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body,
                                                       [0, ignore_mask])  # 对所有图片循环，得到ignore_mask[b][13,13,3]
        ignore_mask = ignore_mask.stack()  # 将一个列表中维度数目为R的张量堆积起来形成维度为R+1的新张量,R应该就是b。
        ignore_mask = K.expand_dims(ignore_mask, -1)  # ignore_mask的shape是(b,13,13,3,1)
        # 当一张box的最大IOU低于ignore_thresh，则作为负样本参与计算confidence 损失。
        # 这里保存的应该是iou满足条件的BBOX

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        # 如果该位置本来有框，那么计算1与置信度的交叉熵
        # 如果该位置本来没有框，而且满足best_iou<ignore_thresh，则被认定为负样本
        # best_iou<ignore_thresh用于限制负样本数量
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='loss: ')
    return loss
