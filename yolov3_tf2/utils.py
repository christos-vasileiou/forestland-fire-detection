from absl import app, logging
import numpy as np
import tensorflow as tf
import cv2
import time
import os
from yolov3_tf2.dataset import transform_images
from IPython.display import Image, display

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]

def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not (layer.name.startswith('conv2d') or \
                    layer.name.endswith('conv2d') ):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    (sub_model.layers[i + 1].name.startswith('batch_norm') or \
                        sub_model.layers[i + 1].name.endswith('batch_norm') ):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()

def calc_iou(box_1, box_2, metrics=None):
    """
    --- Input
    - box_1: predicted box
    - box_2: true_box
    --- Output
    - intersection, Box_1_area
    - intersection, Box_2_area
    - iou
    """
    int_w = tf.maximum(tf.minimum(box_1[...,2], box_2[...,2])-tf.maximum(box_1[...,0], box_2[...,0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[...,3], box_2[...,3])-tf.maximum(box_1[...,1], box_2[...,1]), 0)
    intersection = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    if metrics == 'precision': 
        return tf.reshape(intersection, (-1,1)), tf.reshape(box_1_area, (-1,1)) # precision defined as intersection over area of predicted box
    elif metrics == 'recall': 
        return tf.reshape(intersection, (-1,1)), tf.reshape(box_2_area, (-1,1)) # precision defined as intersection over area of actual object(=target box)
    else:
        return intersection / (box_1_area + box_2_area - intersection)


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))
    
    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2) # predicted box
    box_2 = tf.expand_dims(box_2, 0) # target box
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)
    iou = calc_iou(box_1, box_2, metrics=None)
    return iou
    
# 1 - FONT_HERSHEY_SIMPLEX, 
# 2 - FONT_HERSHEY_PLAIN, 
# 3 - FONT_HERSHEY_DUPLEX, 
# 4 - FONT_HERSHEY_COMPLEX, 
# 5 - FONT_HERSHEY_TRIPLEX, 
# 6 - FONT_HERSHEY_COMPLEX_SMALL, 
# 7 - FONT_HERSHEY_SCRIPT_SIMPLEX,
# 8 - FONT_HERSHEY_SCRIPT_COMPLEX

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

def cast_layer(layer, pretrained_layer):
    print("layer: ", type(layer))
    if isinstance(layer, tf.keras.layers.InputLayer):
        print("Input setup")
        layer.set_weights(                          
            tf.cast(
                pretrained_layer.get_weights(),
                layer.get_config()['dtype'] ) )
    elif isinstance(layer, tf.keras.layers.Lambda):
        print("\n...Lambda Layer...\n")
    elif isinstance(layer, tf.keras.layers.Conv2D):
        print("Conv2D setup")
        if not layer.get_config()['use_bias']:
            print("use_bias:", layer.get_config()['use_bias'])
            layer.set_weights(                          
                tf.cast(
                    pretrained_layer.get_weights(),
                    layer.get_config()['dtype'] ) )
    elif isinstance(layer, tf.keras.Model):
        for l in layer.layers:
            print("next: ", type(l), " name: ", l.name, " -> ", type(pretrained_layer.get_layer(l.name)) )
            cast_layer(l, pretrained_layer.get_layer(l.name))    

def print_all_layers(model, p_details=False):
    if isinstance(model, tf.keras.Model):
        model.summary(line_length=200)
        if p_details:
            tf.keras.utils.plot_model(model, to_file=model.name+'.png', show_shapes=True)

        for _, l in enumerate(model.layers):
            print_all_layers(l, p_details=p_details)
    else:
        if p_details:
            print("{}, {}".format(model.name, model.get_config()))


def Inference(yolo, classes, image, size=416, scale=1.):
    import IPython
    with open(classes) as f:
        class_names = [c.strip() for c in f.readlines()]
    logging.info('classes loaded')

    with open(image, 'rb') as f2:
        img_raw = tf.io.decode_image(f2.read(), channels=3)
        if len(tf.shape(img_raw)) == 3:
            width = np.array(tf.shape(img_raw))[1]
        else:
            width = np.array(tf.shape(img_raw))[2]
        img = tf.expand_dims(img_raw, 0)
        img_x = transform_images(img, size)
    start = time.perf_counter()
    boxes, scores, classes, nums = yolo(img_x)
    inference_time = str((time.perf_counter() - start) * 1000)
    logging.info('time: {0:9.4} ms'.format(inference_time))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info("\t{}, {}, {}".format(class_names[int(classes[0][i])],
                                        scores[0][i].numpy(),
                                        boxes[0][i].numpy()))
    
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    output_scale = int(np.array(scale*width))
    ipython = IPython.get_ipython()
    if not ipython:
        try:
            import google.colab.load_ipython_extension as load_ipython_extension
            load_ipython_extension(ipython)
        except:
            logging.info("To be shown an image you need to use Colab Notebook")
    logging.info("Interactive Python is active: {}".format(
        IPython.get_ipython))
    display(Image(data=bytes(cv2.imencode('.jpg', img)[1]), width=output_scale, embed=True))
    cv2.imwrite('./output.jpg', img)

    logging.info("\n`output.jpg` extracted and it has shown up!")

def setup_accelerator(accelerator):
    strategy = None
    if 'GPU' in accelerator:
        if 'COLAB_GPU' in os.environ:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            for physical_device in physical_devices:
                if not tf.config.experimental.get_memory_growth(physical_device):
                    tf.config.experimental.set_memory_growth(physical_device, True)
            tf.config.set_visible_devices(physical_devices, 'GPU')
            logical_devices = tf.config.list_logical_devices('GPU')
            assert len(logical_devices) == len(physical_devices)
            logging.info("Synchronous Execution: {}".format(tf.config.experimental.get_synchronous_execution()))
            logging.info('--- GPU is ready! ---')
        else:
            logging.info('--- GPU is NOT connected! ---')
    elif 'TPU' in accelerator:
        if 'COLAB_TPU_ADDR' in os.environ:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://'+os.environ['COLAB_TPU_ADDR'])
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.experimental.TPUStrategy(resolver)
            logging.info('--- TPU is ready! ---')
        else:
            logging.info('--- TPU is NOT connected! ---')
    elif 'none' in accelerator:
        logging.info('CPU is running...')

    return strategy
