from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, init_ops
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    ReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
    Reshape
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.mixed_precision import experimental as tf_mixed_precision
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy,
    categorical_crossentropy,
    Loss
)
from .utils import (
    broadcast_iou, 
    calc_iou
)

flags.DEFINE_integer('yolo_max_boxes', 100,
                     'maximum number of boxes per image')
flags.DEFINE_float('yolo_iou_threshold', 0.5, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.5, 'score threshold')
flags.DEFINE_boolean('edge_tpu', False, 'Create model using edge_tpu`s specification')
flags.DEFINE_string('anchorstxt', './data/anchors.txt', 'Using Anchors based on fire-smoke dataset: boolean[True|False]')

yolo_anchors = np.loadtxt('./my_data/anchors.txt', dtype=np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.loadtxt('./my_data/anchors_tiny.txt', dtype=np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])

"""
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), # 3rd output 52x52 grid
                         (30, 61), (62, 45), (59, 119), # 2nd output 26x6 grid
                         (116, 90), (156, 198), (373, 326)], # 1st output 13x13 grid
                        np.float32) / 416

yolo_anchors_fire =  np.array([(32, 30), (47, 71), (55, 119), 
                               (106, 82), (68, 174), (131, 154), 
                               (107, 245), (187, 295), (331, 208)],
                                 np.float32) / 416
"""

def adjust_yolo_anchors(anchors, size=416):
    # Anchors are written above are matching perfectly for size 416. Adjusting them in other sizes, 
    # ratio is specified.
    if size is not None:
        ratio = size / 416
        return anchors*ratio
    return anchors


def DarknetConv(x, filters, size, strides=1, batch_norm=True, act_fn='leaky', name=None):
    cnt=0
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)), name=name+'_zeropad_'+str(cnt))(x)  # top left half-padding
        cnt+=1
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005), name=name+'_'+str(cnt)+'_conv2d')(x)
    if batch_norm:
        x = BatchNormalization(name=name+'_batch_norm')(x)
        if act_fn == 'relu':
            x = ReLU(name=name+'_ReLU')(x)
        elif act_fn == 'leaky':
            x = LeakyReLU(alpha=0.1, name=name+'_LeakyReLU')(x)
    return x


def DarknetResidual(x, filters, act_fn='leaky', name=None):
    cnt=0
    prev = x
    x = DarknetConv(x, filters=filters // 2, size=1, act_fn=act_fn, name=name+'_'+str(cnt))
    cnt+=1
    x = DarknetConv(x, filters=filters, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    x = Add(name=name+'_Add')([prev, x])
    return x


def DarknetBlock(x, filters, blocks, act_fn='leaky', name=None):
    cnt=0
    x = DarknetConv(x, filters=filters, size=3, strides=2, act_fn=act_fn, name=name+'_'+str(cnt))
    for _ in range(blocks):
        cnt+=1
        x = DarknetResidual(x, filters=filters, act_fn=act_fn, name=name+'_'+str(cnt))
    return x


def Darknet(size=None, act_fn='leaky', name=None, batch_size=None):
    cnt=0
    x = inputs = Input(shape=[size, size, 3], batch_size=batch_size, name=name+'_input')
    x = DarknetConv(x, filters=32, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    cnt+=1
    x = DarknetBlock(x, filters=64, blocks=1, act_fn=act_fn, name=name+'_'+str(cnt))
    cnt+=1
    x = DarknetBlock(x, filters=128, blocks=2, act_fn=act_fn, name=name+'_'+str(cnt))  # skip connection
    cnt+=1
    x = x_36 = DarknetBlock(x, filters=256, blocks=8, act_fn=act_fn, name=name+'_'+str(cnt))  # skip connection
    cnt+=1
    x = x_61 = DarknetBlock(x, filters=512, blocks=8, act_fn=act_fn, name=name+'_'+str(cnt))
    cnt+=1
    x = DarknetBlock(x, filters=1024, blocks=4, act_fn=act_fn, name=name+'_'+str(cnt))
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)

def DarknetTiny(size=None, act_fn='leaky', name=None, batch_size=None):
    cnt=0
    x = inputs = Input(shape=[size, size, 3], batch_size=batch_size, name=name+'_input')
    x = DarknetConv(x, filters=16, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name=name+'_MaxPool_'+str(cnt))(x)
    cnt+=1
    x = DarknetConv(x, filters=32, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name=name+'_MaxPool_'+str(cnt))(x)
    cnt+=1
    x = DarknetConv(x, filters=64, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name=name+'_MaxPool_'+str(cnt))(x)
    cnt+=1
    x = DarknetConv(x, filters=128, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name=name+'_MaxPool_'+str(cnt))(x)
    cnt+=1
    x = x_8 = DarknetConv(x, filters=256, size=3, act_fn=act_fn, name=name+'_'+str(cnt))  # skip connection
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name=name+'_MaxPool_'+str(cnt))(x)
    cnt+=1
    x = DarknetConv(x, filters=512, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    x = MaxPool2D(pool_size=2, strides=1, padding='same', name=name+'_MaxPool_'+str(cnt))(x)
    cnt+=1
    x = DarknetConv(x, filters=1024, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    return tf.keras.Model(inputs, (x_8, x), name=name)

def YoloConv(filters, act_fn='leaky', name=None, batch_size=None):
    def yolo_conv(x_in):
        cnt=0
        if isinstance(x_in, tuple):
            inputs = Input(shape=x_in[0].shape[1:], batch_size=batch_size, name=name+'_input_0'), Input(shape=x_in[1].shape[1:], batch_size=batch_size, name=name+'_input_1')
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1, act_fn=act_fn, name=name+'_'+str(cnt))
            cnt+=1
            x = UpSampling2D(2, name=name+'_UpSampling_nearest', interpolation='nearest')(x)
            x = Concatenate(name=name+'_Concatenate')([x, x_skip])
        else:
            x = inputs = Input(shape=x_in.shape[1:], batch_size=batch_size, name=name+'_input')
        x = DarknetConv(x, filters=filters, size=1, act_fn=act_fn, name=name+'_'+str(cnt))
        cnt+=1
        x = DarknetConv(x, filters=filters*2, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
        cnt+=1
        x = DarknetConv(x, filters=filters, size=1, act_fn=act_fn, name=name+'_'+str(cnt))
        cnt+=1
        x = DarknetConv(x, filters=filters*2, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
        cnt+=1
        x = DarknetConv(x, filters=filters, size=1, act_fn=act_fn, name=name+'_'+str(cnt))
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv

def YoloConvTiny(filters, act_fn='leaky', name=None, batch_size=None):
    def yolo_conv(x_in):
        cnt=0
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:], batch_size=batch_size, name=name+'_input_0'), Input(x_in[1].shape[1:], batch_size=batch_size, name=name+'_input_1')
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters=filters, size=1, act_fn=act_fn, name=name+'_'+str(cnt))
            cnt+=1
            x = UpSampling2D(2, name=name+'_UpSampling_nearest', interpolation='nearest')(x)
            x = Concatenate(name=name+'_Concatenate')([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:], batch_size=batch_size, name=name+'_input')
            x = DarknetConv(x, filters=filters, size=1, act_fn=act_fn, name=name+'_'+str(cnt))

        return Model(inputs, x, name=name)(x_in)
    return yolo_conv

def YoloOutput(filters, anchors, classes, batch_size=None, act_fn='leaky', name=None):
    def yolo_output(x_in):
        cnt=0
        x = inputs = Input(shape=x_in.shape[1:], batch_size=batch_size, name=name+'_input')
        x = DarknetConv(x, filters=filters * 2, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
        cnt+=1
        x = DarknetConv(x, filters=anchors * (classes + 5), size=1, batch_norm=False, act_fn='linear', name=name+'_'+str(cnt))
        x = Reshape((x.get_shape()[1], x.get_shape()[2], anchors, classes+5), batch_size=batch_size, name=name+'_reshape')(x)
        #x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)), name=name+'_reshape')(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output


def yolo_boxes(pred, anchors, classes, calc_loss=False):
    """
    - input:
    --- pred: is one of the yolo's outputs. Size (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
            e.g. if input image is 416x416 then grid is downscaled by 32, 16, 8.
             with size 13x13, 26x26, 52x52 respectively.
    --- anchors: are the 3 of the 9 anchors that are used.
             They have been assigned the corresponding anchors using anchor_mask
             according the downscale.
    --- classes: a list of classes.
    
    - outputs:
    --- bbox: are bounding boxes with size (1, downscaledGrid, downscaledGrid, (x1,y1,x2,y2))
            bounding boxes are defined by top-left point (x1,y1) 
            and bottom-right (x2,y2) using the following equations. 
            Let assume that (x,y) is the center, (w,h) is the width-height of bbox
            and (cx,cy) the top-left cell's coordinates of the grid :
            1. xy = sigmoid(pred_xy)+cxcy
            2. wh = anchors * e^pred_wh
            3. x1y1 = xy - wh/2
            4. x2y2 = xy + wh/2
            5. bbox = [x1y1, x2y2]
    --- objectness: is probability of objectness.
            if there is any possible class.
            objectness = sigmoid(objectness)
    --- class_probs: is the probability of each class.
            class_probs = sigmoid(class_probs)
    --- pred_box: is the box that it's responsible for yolo-loss calculation.
            Consist of:
            pred_box = [sigmoid(pred_xy), pred_wh]
    """
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)
    #tf.print("network output xy: \n{}\n".format(box_xy[0,7,7]))
    #tf.print("network output anchor shape: \n{}\n".format(box_xy[0,7,7].get_shape()))

    box_xy = tf.sigmoid(box_xy)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss
    class_probs = tf.sigmoid(class_probs)

    if calc_loss is False:
        objectness = tf.sigmoid(objectness)
        #class_probs = tf.sigmoid(class_probs)
    

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    #tf.print("grid: \n{}".format(grid[7,7]))
    #tf.print("grid: \n{}".format(grid[7,8]))
    #tf.print("grid_size: \n{}\n".format(grid_size))
    
    box_xy = (tf.cast(box_xy, tf.float32) + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = anchors * tf.exp(box_wh)

    #tf.print("Considering grid: box_xy: \n{}".format(box_xy[0,7,7]))
    #tf.print("Considering anchors: box_wh: \n{}\n".format(box_wh[0,7,7]))

    box_x1y1 = tf.cast(box_xy, tf.float32) - tf.cast(box_wh, tf.float32) / 2
    box_x2y2 = tf.cast(box_xy, tf.float32) + tf.cast(box_wh, tf.float32) / 2
    #tf.print("p1(x1,y1): \n{}".format(box_x1y1[0,7,7]))
    #tf.print("p2(x2,y2): \n{}\n".format(box_x2y2[0,7,7]))
    
    bbox = tf.concat([tf.cast(box_x1y1, tf.float32), tf.cast(box_x2y2, tf.float32)], axis=-1)
    #tf.print("bbox: \n{}".format(bbox[0,7,7]))
    #tf.print("pred_box: \n{}\n".format(pred_box[0,7,7]))
    #tf.print("objectness: \n{}".format(objectness[0,7,7]))
    #tf.print("class_probs: \n{}".format(class_probs[0,7,7]))
    
    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, class probability
    box, confidence, class_probs = [], [], []
    
    for o in outputs:
        box.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        confidence.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        class_probs.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(box, axis=1)
    confidence = tf.concat(confidence, axis=1)
    class_probs = tf.concat(class_probs, axis=1)
    
    if classes > 1:
        scores = confidence * class_probs
    else:
        scores = confidence
            
    boxes = tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4))
    scores = tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1]))
    tf.print("boxes: {}, scores; {}".format(boxes.get_shape(), scores.get_shape()))
    """
    tf.print("score threshold: \n{}".format(FLAGS.yolo_score_threshold))
    tf.print("MASK: \n{}".format(scores>FLAGS.yolo_score_threshold))
    tf.print("boolean_mask: \n{}".format(tf.boolean_mask(scores, tf.cast(scores>FLAGS.yolo_score_threshold, tf.bool))))
    nms_index = tf.image.non_max_suppression(
        boxes = bbox,
        scores = scores,
        max_output_size = FLAGS.yolo_max_boxes,
        iou_threshold = FLAGS.yolo_iou_threshold,
        score_threshold = FLAGS.yolo_score_threshold
    )
    boxes = tf.gather(bbox, nms_index)
    scores = tf.gather(scores, nms_index)
    classes = tf.gather(classes, nms_index)
    valid_detections = tf.size(nms_index)
    """
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )
    """
    tf.print("boxes: \n{}".format(boxes))
    tf.print("scores: \n{}".format(scores))
    tf.print("valid_detections: \n{}".format(valid_detections))
    """
    return boxes, scores, classes, valid_detections


def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False, 
           mixed_precision='float32', batch_size=None):
    
    if mixed_precision=='float16':
        policy = tf_mixed_precision.Policy('mixed_'+mixed_precision)
        tf_mixed_precision.set_policy(policy)
    elif mixed_precision=='bfloat16':
        policy = tf_mixed_precision.Policy('mixed_'+mixed_precision)
        tf_mixed_precision.set_policy(policy)
    else:
        tf.keras.backend.set_floatx('float32')
    if FLAGS.edge_tpu:
        act_fn = 'relu'
    else:
        act_fn = 'leaky'
    x = inputs = Input(shape=[size, size, channels], batch_size=batch_size, name='input')
        
    x_36, x_61, x = Darknet(size=size, batch_size=batch_size, act_fn=act_fn, name='yolo_darknet')(x)

    x = YoloConv(512, batch_size=batch_size, act_fn=act_fn, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, batch_size=batch_size, act_fn=act_fn, name='yolo_output_0')(x)

    x = YoloConv(256, batch_size=batch_size, act_fn=act_fn, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, batch_size=batch_size, act_fn=act_fn, name='yolo_output_1')(x)

    x = YoloConv(128, batch_size=batch_size, act_fn=act_fn, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, batch_size=batch_size, act_fn=act_fn, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3', trainable=training)
    
    anchors = adjust_yolo_anchors(anchors, size)
    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0', dtype='float32')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1', dtype='float32')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2', dtype='float32')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms', dtype='float32')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3', trainable=training)

def yolo_v3_tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, training=False,
               mixed_precision='float32', batch_size=None):
    if mixed_precision=='float16':
        policy = tf_mixed_precision.Policy('mixed_'+mixed_precision)
        tf_mixed_precision.set_policy(policy)
    elif mixed_precision=='bfloat16':
        policy = tf_mixed_precision.Policy('mixed_'+mixed_precision)
        tf_mixed_precision.set_policy(policy)
    else:
        tf.keras.backend.set_floatx('float32')

    if FLAGS.edge_tpu:
        act_fn = 'relu'
    else:
        act_fn = 'leaky'

    x = inputs = Input([size, size, channels], batch_size=batch_size, name='input')

    # DarknetTiny
    name='yolo_darknet'
    cnt=0
    x = DarknetConv(x, filters=16, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name=name+'_MaxPool_'+str(cnt))(x)
    cnt+=1
    x = DarknetConv(x, filters=32, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name=name+'_MaxPool_'+str(cnt))(x)
    cnt+=1
    x = DarknetConv(x, filters=64, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name=name+'_MaxPool_'+str(cnt))(x)
    cnt+=1
    x = DarknetConv(x, filters=128, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name=name+'_MaxPool_'+str(cnt))(x)
    cnt+=1
    x = x_8 = DarknetConv(x, filters=256, size=3, act_fn=act_fn, name=name+'_'+str(cnt))  # skip connection
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name=name+'_MaxPool_'+str(cnt))(x)
    cnt+=1
    x = DarknetConv(x, filters=512, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    x = MaxPool2D(pool_size=2, strides=1, padding='same', name=name+'_MaxPool_'+str(cnt))(x)
    cnt+=1
    x = DarknetConv(x, filters=1024, size=3, act_fn=act_fn, name=name+'_'+str(cnt))

    # Yolo Conv Net 0
    name='yolo_conv_0'
    cnt=0
    x = x_13 = DarknetConv(x, filters=256, size=1, act_fn=act_fn, name=name+'_'+str(cnt))

    # Yolo Output Net 0
    name='yolo_output_0'
    cnt=0
    x = DarknetConv(x, filters=256 * 2, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    cnt+=1
    x = DarknetConv(x, filters=len(masks[0]) * (classes + 5), size=1, batch_norm=False, act_fn='linear', name=name+'_'+str(cnt))
    output_0 = Reshape((x.get_shape()[1], x.get_shape()[2], len(masks[0]), classes+5), batch_size=batch_size, name=name+'_reshape')(x)

    # Yolo Conv Net 1
    name='yolo_conv_1'
    x = x_13
    x_skip = x_8
    # concat with skip connection
    x = DarknetConv(x, filters=128, size=1, act_fn=act_fn, name=name+'_'+str(cnt))
    cnt+=1
    x = UpSampling2D(2, name=name+'_UpSampling_nearest', interpolation='nearest')(x)
    x = Concatenate(name=name+'_Concatenate')([x, x_skip])

    # Yolo Output Net 1
    name='yolo_output_1'
    cnt=0
    x = DarknetConv(x, filters=128 * 2, size=3, act_fn=act_fn, name=name+'_'+str(cnt))
    cnt+=1
    x = DarknetConv(x, filters=len(masks[1]) * (classes + 5), size=1, batch_norm=False, act_fn='linear', name=name+'_'+str(cnt))
    output_1 = Reshape((x.get_shape()[1], x.get_shape()[2], len(masks[1]), classes+5), batch_size=batch_size, name=name+'_reshape')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3_tiny')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))

    return Model(inputs, outputs, name='yolov3_tiny')



def YoloV3Tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, training=False,
               mixed_precision='float32', batch_size=None):
    
    if mixed_precision=='float16':
        policy = tf_mixed_precision.Policy('mixed_'+mixed_precision)
        tf_mixed_precision.set_policy(policy)
    elif mixed_precision=='bfloat16':
        policy = tf_mixed_precision.Policy('mixed_'+mixed_precision)
        tf_mixed_precision.set_policy(policy)
    else:
        tf.keras.backend.set_floatx('float32')

    if FLAGS.edge_tpu:
        act_fn = 'relu'
    else:
        act_fn = 'leaky'

    x = inputs = Input([size, size, channels], batch_size=batch_size, name='input')
    x_8, x = DarknetTiny(size=size, batch_size=batch_size, act_fn=act_fn, name='yolo_darknet')(x)

    x = YoloConvTiny(256, batch_size=batch_size, act_fn=act_fn, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, batch_size=batch_size, act_fn=act_fn, name='yolo_output_0')(x)

    x = YoloConvTiny(128, batch_size=batch_size, act_fn=act_fn, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, batch_size=batch_size, act_fn=act_fn, name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3_tiny')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='yolov3_tiny')

class experimentalYoloPrecision(tf.keras.metrics.Precision):
    
    def __init__(self,
                 anchors=None,
                 num_classes=None,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 dtype=None,
                 name=None
                 #**kwargs
                 ):
        super(experimentalYoloPrecision, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id
        self.anchors = anchors
        self.num_classes = num_classes
        self.thresholds = 0.5 if thresholds is None else thresholds
        
        self.mAP = self.add_weight(
            'mAP', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, _ = yolo_boxes(y_pred, self.anchors, self.num_classes, calc_loss=False)
        #tf.print("pred_scores: {}".format(pred_scores.get_shape()))
        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4,1,1), axis=-1)
        
        # convert to one-hot encoding class prediction and hold only those whose target objectivity has been assigned to 1
        # moreover, do the same for score confidence.
        # TODO: WRITE correct comments

        # 3. calculate masks, find true and false positives and calculate averageprecision per class
        obj_mask = tf.squeeze(true_obj, -1)
        #tf.print("obj_mask: \n{}".format(obj_mask.get_shape()))
        #best_iou = tf.math.reduce_max(broadcast_iou(pred_box, tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))), axis=-1), 
        best_iou = tf.map_fn(
                            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1), 
                            (tf.cast(pred_box, tf.float32), tf.cast(true_box, tf.float32), tf.cast(obj_mask, tf.float32)), 
                            dtype=tf.float32)

        pred_score_confidence = pred_obj*pred_class
        pred_score_confidence_idx = tf.expand_dims(tf.argmax(pred_score_confidence, axis=-1), axis=-1)

        mAP = 0
        obj_mask = tf.expand_dims(obj_mask, axis=-1)
        cl = 0
        for c in range(self.num_classes):
            cl += 1
            Precision = 0.
            for thr in self.thresholds:
                class_mask = tf.squeeze(tf.logical_and(true_class_idx == c, tf.cast(obj_mask, tf.bool)), axis=-1)
                pred_class_mask = tf.squeeze(tf.logical_and(pred_score_confidence_idx == c, tf.cast(obj_mask, tf.bool)), axis=-1)
                true_idx = class_mask == pred_class_mask

                best_iou_per_class = tf.boolean_mask(best_iou, class_mask)
                #best_iou_per_class = tf.boolean_mask(best_iou, true_idx)
                
                tp = tf.cast(tf.reduce_sum(tf.cast(best_iou_per_class >= thr, tf.int32)), tf.float32)
                fp = tf.cast(tf.reduce_sum(tf.cast(best_iou_per_class < thr, tf.int32)), tf.float32)
                if tf.math.greater(tf.math.add(tp,fp), 0):
                    Precision += tp/(tp+fp)
            APrecision = Precision / len(self.thresholds)
            if isinstance(Precision, float):
                cl -= 1
                continue
            APrecision = tf.where(tf.math.is_nan(APrecision), tf.zeros_like(APrecision, dtype=APrecision.dtype), APrecision)
            #tf.print("AP: {0:.3}%".format(APrecision*100))
            mAP += APrecision
        if cl != 0:
            mAP /= cl
        #tf.print("\nbest_iou: {0}\nmAP: {1:.5}%\ntrue_class_idx: {2}\n".format(best_iou.get_shape(), mAP*100, true_class_idx.get_shape()))
        #tf.print("mAP: {0:.5}".format(mAP*100), )
        self.mAP.assign(tf.cast(mAP, tf.float32))

    def result(self):
        return self.mAP

    def reset_states(self):
        self.mAP.assign(0.0)

class experimentalYoloRecall(tf.keras.metrics.Recall):
    
    def __init__(self,
                 anchors=None,
                 num_classes=None,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 dtype=None,
                 name=None
                 #**kwargs
                 ):
        super(experimentalYoloRecall, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id
        self.anchors = anchors
        self.num_classes = num_classes
        self.thresholds = 0.5 if thresholds is None else thresholds
        
        self.Recall = self.add_weight(
            'Recall', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, _ = yolo_boxes(y_pred, self.anchors, self.num_classes, calc_loss=False)
        #tf.print("pred_scores: {}".format(pred_scores.get_shape()))
        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4,1,1), axis=-1)
        
        # convert to one-hot encoding class prediction and hold only those whose target objectivity has been assigned to 1
        # moreover, do the same for score confidence.
        # TODO: WRITE correct comments

        # 3. calculate masks, find true and false positives and calculate averageprecision per class
        obj_mask = tf.squeeze(true_obj, -1)
        #tf.print("obj_mask: \n{}".format(obj_mask.get_shape()))
        #best_iou = tf.math.reduce_max(broadcast_iou(pred_box, tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))), axis=-1), 
        best_iou = tf.map_fn(
                            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1), 
                            (tf.cast(pred_box, tf.float32), tf.cast(true_box, tf.float32), tf.cast(obj_mask, tf.float32)), 
                            dtype=tf.float32)

        pred_score_confidence = pred_obj*pred_class
        pred_score_confidence_idx = tf.expand_dims(tf.argmax(pred_score_confidence, axis=-1), axis=-1)

        Recall = 0
        obj_mask = tf.expand_dims(obj_mask, axis=-1)
        cl = 0
        for c in range(self.num_classes):
            cl += 1
            Recall = 0.
            for thr in self.thresholds:
                class_mask = tf.squeeze(tf.logical_and(true_class_idx == c, tf.cast(obj_mask, tf.bool)), axis=-1)
                pred_class_mask = tf.squeeze(tf.logical_and(pred_score_confidence_idx == c, tf.cast(obj_mask, tf.bool)), axis=-1)
                true_idx = class_mask == pred_class_mask

                best_iou_per_class = tf.boolean_mask(best_iou, class_mask)
                #best_iou_per_class = tf.boolean_mask(best_iou, true_idx)
                
                tp = tf.cast(tf.reduce_sum(tf.cast(best_iou_per_class >= thr, tf.int32)), tf.float32)
                ground_truths = tf.cast(tf.reduce_sum(tf.cast(obj_mask, tf.int32)), tf.float32)
                if tf.math.greater(ground_truths, 0):
                    Recall += tp/ground_truths
            ARecall = Recall / len(self.thresholds)
            if isinstance(Recall, float):
                cl -= 1
                continue
            ARecall = tf.where(tf.math.is_nan(ARecall), tf.zeros_like(ARecall, dtype=ARecall.dtype), ARecall)
            #tf.print("AP: {0:.3}%".format(APrecision*100))
            Recall += ARecall
        if cl != 0:
            Recall /= cl
        #tf.print("\nbest_iou: {0}\nmAP: {1:.5}%\ntrue_class_idx: {2}\n".format(best_iou.get_shape(), mAP*100, true_class_idx.get_shape()))
        #tf.print("mAP: {0:.5}".format(mAP*100), )
        self.Recall.assign(tf.cast(Recall, tf.float32))

    def result(self):
        return self.Recall

    def reset_states(self):
        self.Recall.assign(0.0)

def square_loss(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return tf.math.square(y_pred - y_true)


def YoloLoss(anchors, classes=80, ignore_thresh=0.5, dtype='float32'):
    def yolo_loss(y_true, y_pred):

        # parameters to scale the loss.
        tfdtype = tf.float32
        if dtype=='float16':
            tfdtype = tf.float16
        obj_scale = 5
        class_scale = 5

        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes, calc_loss=True)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]
        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = tf.cast(true_xy, tfdtype) * tf.cast(grid_size, tfdtype) - \
            tf.cast(grid, tfdtype)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        #tf.print("obj_mask: \n{}".format(obj_mask.get_shape()))
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
                            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1), 
                            (tf.cast(pred_box, tfdtype), tf.cast(true_box, tfdtype), tf.cast(obj_mask, tfdtype)), 
                            dtype=tfdtype
                            )

        ignore_mask = tf.cast(best_iou < ignore_thresh, tfdtype)
        #tf.print("pred_class: \n{}".format(pred_class.get_shape()))
        #tf.print("ignore_mask: \n{}".format(ignore_mask.get_shape()))
        
        # 5. calculate all losses

        xy_loss = square_loss(tf.cast(true_xy, tfdtype), tf.cast(pred_xy, tfdtype))
        xy_loss = tf.cast(obj_mask, tfdtype) * tf.cast(box_loss_scale, tfdtype) * tf.reduce_sum(xy_loss, axis=-1)

        wh_loss = square_loss(true_wh, pred_wh)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(wh_loss, axis=-1)
        
        #tf.print("true_obj:\n{}\npred_obj:\n{}".format(true_obj[0,7,7], pred_obj[0,7,7]))
        obj_loss = binary_crossentropy(y_true=true_obj, y_pred=pred_obj, from_logits=True)
        obj_loss = tf.cast(obj_scale, tfdtype) * tf.cast(obj_mask, tfdtype) * tf.cast(obj_loss, tfdtype)
        noobj_loss = tf.cast(obj_scale, tfdtype) * (1 - tf.cast(obj_mask, tfdtype)) * tf.cast(ignore_mask, tfdtype) * tf.cast(obj_loss, tfdtype)
        obj_loss = obj_loss + noobj_loss

        if FLAGS.binary_class_loss:
            true_class_one_hot = tf.squeeze(tf.one_hot(indices=tf.cast(true_class_idx, tf.uint8), depth=classes, on_value=1.0, off_value=0.0, axis=-1), -2)
            #tf.print("true_class_one_hot: {}, {}\npred_class: {}, {}".format(tf.shape(true_class_one_hot), true_class_one_hot.dtype, tf.shape(pred_class), pred_class.dtype))
            for i in range(classes):
                #tf.print("true_class_one_hot[..., {}]: {}, pred_class[..., {}]: {}".format(i, tf.shape(true_class_one_hot[..., i]), i, tf.shape(pred_class[..., i])))
                y_true = tf.expand_dims(true_class_one_hot[..., i], axis=-1)
                y_pred = tf.expand_dims(pred_class[..., i], axis=-1)
                #tf.print("obj_mask: {}, y_true: {}, y_pred: {}".format(obj_mask.get_shape(), y_true.get_shape(), y_pred.get_shape()))
                if i == 0:
                    #class_loss = obj_mask * cl_loss + (1-obj_mask) * ignore_mask * cl_loss
                    bce_cl_loss = binary_crossentropy(y_true=y_true, y_pred=y_pred)
                    temp_class_loss = class_scale * obj_mask * bce_cl_loss
                    temp_noclass_loss = (1-obj_mask) * ignore_mask * bce_cl_loss
                    class_loss = tf.expand_dims(temp_class_loss + temp_noclass_loss, axis=-1)
                else:
                    bce_cl_loss = binary_crossentropy(y_true=y_true, y_pred=y_pred)
                    temp_class_loss = class_scale * obj_mask * bce_cl_loss
                    temp_noclass_loss = (1-obj_mask) * ignore_mask * bce_cl_loss
                    temp_class_loss = tf.expand_dims(temp_class_loss + temp_noclass_loss, axis=-1)
                    class_loss = tf.concat([class_loss, 
                                            temp_class_loss], 
                                            axis=-1)
            class_loss = tf.reduce_sum(class_loss, axis=4)
        else:
            class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)

        #tf.print("\nclass_loss: {}".format(class_loss.get_shape()))
        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(tf.cast(obj_loss, tfdtype), axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
        return tf.cast(xy_loss, tfdtype) + \
                tf.cast(wh_loss, tfdtype) + \
                tf.cast(obj_loss, tfdtype) + \
                tf.cast(class_loss, tfdtype)

    return yolo_loss

#loss: 5.1875 - yolo_output_0_loss: 0.3832 - yolo_output_1_loss: 0.3855 - yolo_output_2_loss: 0.0948 - val_loss: 10.1034 - val_yolo_output_0_loss: 3.6272 - val_yolo_output_1_loss: 1.2216 - val_yolo_output_2_loss: 0.9655 - lr: 0.0010