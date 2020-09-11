from models import yolo_tiny_anchors, yolo_tiny_anchor_masks
from PIL import Image, ImageDraw
from absl import app, logging, flags
from absl.flags import FLAGS
import tflite_runtime.interpreter as tflite
import numpy as np
import platform
import collections
import time
import os

flags.DEFINE_float('score_threshold', 0.25, 'score threshold')
flags.DEFINE_float('iou_threshold', 0.25, 'iou threshold')

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

    @property
    def width(self):
        """Returns bounding box width."""
        return self.xmax - self.xmin

    @property
    def height(self):
        """Returns bounding box height."""
        return self.ymax - self.ymin

    @property
    def area(self):
        """Returns bound box area."""
        return self.width * self.height

    @property
    def valid(self):
        """Returns whether bounding box is valid or not.
        Valid bounding box has xmin <= xmax and ymin <= ymax which is equivalent to
        width >= 0 and height >= 0.
        """
        return self.width >= 0 and self.height >= 0

    def scale(self, sx, sy):
        """Returns scaled bounding box."""
        return BBox(xmin=sx * self.xmin,
                    ymin=sy * self.ymin,
                    xmax=sx * self.xmax,
                    ymax=sy * self.ymax)

    def translate(self, dx, dy):
        """Returns translated bounding box."""
        return BBox(xmin=dx + self.xmin,
                    ymin=dy + self.ymin,
                    xmax=dx + self.xmax,
                    ymax=dy + self.ymax)

    def map(self, f):
        """Returns bounding box modified by applying f for each coordinate."""
        return BBox(xmin=f(self.xmin),
                    ymin=f(self.ymin),
                    xmax=f(self.xmax),
                    ymax=f(self.ymax))

    @staticmethod
    def intersect(a, b):
        """Returns the intersection of two bounding boxes (may be invalid)."""
        return BBox(xmin=max(a.xmin, b.xmin),
                    ymin=max(a.ymin, b.ymin),
                    xmax=min(a.xmax, b.xmax),
                    ymax=min(a.ymax, b.ymax))

    @staticmethod
    def union(a, b):
        """Returns the union of two bounding boxes (always valid)."""
        return BBox(xmin=min(a.xmin, b.xmin),
                    ymin=min(a.ymin, b.ymin),
                    xmax=max(a.xmax, b.xmax),
                    ymax=max(a.ymax, b.ymax))

    @staticmethod
    def iou(a, b):
        """Returns intersection-over-union value."""
        intersection = BBox.intersect(a, b)
        if not intersection.valid:
            return 0.0
        area = intersection.area
        return area / (a.area + b.area - area)

def get_interpreter_input_details(interpreter):
    input_details = interpreter.get_input_details()
    return input_details

def get_interpreter_output_details(interpreter):
    output_details = interpreter.get_output_details()
    return output_details

def set_input(interpreter, image):
  """Copies a resized and properly zero-padded image to the input tensor.
  Args:
    interpreter: Interpreter object.
    size: original image size as (width, height) tuple.
    resize: a function that takes a (width, height) tuple, and returns an RGB
      image resized to those dimensions.
  Returns:
    Actual resize ratio, which should be passed to `get_output` function.
  """
  input_details = get_interpreter_input_details(interpreter)
  batch, height, width, channels = input_details[0]['shape']
  h, w = image.size
  print("\n\ninput_details: \n    {}".format(input_details))
  image = image.resize((height, width), resample=Image.BILINEAR)
  nimage = np.array(image)
  scale, zero_point = input_details[0]['quantization']
  nimage[:,:] = nimage / scale + zero_point
  nimage = np.expand_dims(nimage, 0).astype(input_details[0]['dtype'])
  interpreter.set_tensor(input_details[0]['index'], nimage)
  scaleH, scaleW = h/height, w/width
  return (scaleW, scaleH)

def load_labels(path, encoding='utf-8'):
    """Loads labels from file (with or without index numbers).
    Args:
        path: path to label file.
        encoding: label file encoding.
    Returns:
        Dictionary mapping indices to labels.
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}

def sigmoid(x):
    return 1/(1+np.exp(-x))

def np_yolo_boxes(pred, anchors, classes, calc_loss=False):
    grid_size = np.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs, _ = np.split(
        pred, (2, 4, 5, 5+classes), axis=-1)
    box_xy = box_xy[:,:,:,:].astype(np.float128)
    box_wh = box_wh[:,:,:,:].astype(np.float128)
    objectness = objectness[:,:,:,:].astype(np.float128)
    class_probs = class_probs[:,:,:,:].astype(np.float128)
    box_xy = sigmoid(box_xy)
    box_wh = box_wh
    pred_box = np.concatenate((box_xy, box_wh), axis=-1)  # original xywh for loss
    class_probs = sigmoid(class_probs)

    if calc_loss is False:
        objectness = sigmoid(objectness)
    # !!! grid[x][y] == (y, x)
    grid = np.array(np.meshgrid(np.linspace(0, grid_size-1, grid_size), np.linspace(0, grid_size-1, grid_size))).astype('float32')
    grid = np.expand_dims(np.stack([grid[0], grid[1]], axis=-1), axis=2) # [gx, gy, 1, 2]

    box_xy = (box_xy.astype('float32') + grid) / float(grid_size)
    box_wh = anchors * np.exp(box_wh)

    box_x1y1 = box_xy.astype('float32') - box_wh.astype('float32') / 2
    box_x2y2 = box_xy.astype('float32') + box_wh.astype('float32') / 2

    bbox = np.concatenate([box_x1y1.astype('float32'), box_x2y2.astype('float32')], axis=-1)
    return bbox, objectness, class_probs, pred_box

def get_interpreter_output(interpreter, quantization=True):
    output_details = get_interpreter_output_details(interpreter)
    print("\n> 1st output_details:\n    {}\n> 2nd ouput_details:\n    {}".format(output_details[0], output_details[1]))
    outputs = []
    for output in output_details:
        prediction = interpreter.get_tensor(output['index'])
        if quantization:
            o_scale, o_zero = output['quantization']
            prediction = (prediction.astype(np.float32) - o_zero) * o_scale
        outputs.append(prediction)
    return outputs

def get_output(interpreter, image_scale=(1.0, 1.0), quantization=True):
    """Returns list of detected objects."""
    outputs = get_interpreter_output(interpreter, quantization=True)
    anchors = yolo_tiny_anchors
    anchor_masks = yolo_tiny_anchor_masks
    box, confidence, class_probs = [], [], []

    inference_time = []
    # 1st output
    start = time.perf_counter()
    pred_box, pred_obj, pred_class, _ = np_yolo_boxes(outputs[0], anchors[anchor_masks[0]], classes=FLAGS.num_classes)
    inference_time.append(time.perf_counter() - start)
    box.append(np.reshape(pred_box, (-1, np.shape(pred_box)[-1])))
    confidence.append(np.reshape(pred_obj, (-1, np.shape(pred_obj)[-1])))
    class_probs.append(np.reshape(pred_class, (-1, np.shape(pred_class)[-1])))

    # 2nd output
    start = time.perf_counter()
    pred_box, pred_obj, pred_class, _ = np_yolo_boxes(outputs[1], anchors[anchor_masks[1]], classes=FLAGS.num_classes)
    inference_time.append(time.perf_counter() - start)
    box.append(np.reshape(pred_box, (-1, np.shape(pred_box)[-1])))
    confidence.append(np.reshape(pred_obj, (-1, np.shape(pred_obj)[-1])))
    class_probs.append(np.reshape(pred_class, (-1, np.shape(pred_class)[-1])))

    start = time.perf_counter()
    bbox = np.column_stack([[b] for b in box])
    confidence = np.column_stack([[c] for c in confidence])
    class_probs = np.column_stack([[c] for c in class_probs])

    scores = confidence*10 * class_probs*10
    #scores = np.reshape(scores, (np.shape(scores)[0], -1, np.shape(scores)[-1]))
    #boxes = np.reshape(bbox, (np.shape(bbox)[0], -1, 1, 4))
    scores = np.reshape(scores, (-1, np.shape(scores)[-1]))
    boxes = np.reshape(bbox, (-1, 4))

    picked_boxes, picked_class = combined_non_max_suppression(boxes, scores)
    #print("amax: {}\n{}".format(np.amax(scores[0][0]), scores[0][0]))

    def make_object(b, c):
        xmin, ymin, xmax, ymax = boxes[b]
        return Object(
            id=c,
            score=scores[b][c],
            bbox=BBox(xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax))
    objs = [make_object(b, c) for b, c in zip(picked_boxes, picked_class)]
    inference_time.append(time.perf_counter() - start)

    return objs, inference_time

def combined_non_max_suppression(boxes, scores, max_outputs=20):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes are integers, convert them to floats
    # It's very important since it's going to divided.
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of indexes
    picked_box = []
    picked_class = []
    # take the coordinates of the boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # scores[each] = pc * [c0, c1, c2, ...]
    # --- class_bbox: take the index of maximum element of
    #                 scores[each], as a result it defines
    #                 the class with highest probability of
    #                 each bounding box
    # --- max_scores: hold the maximum element of scores[each]
    # --- order: contains in descending order the indexes
    #                       throughout the max_scores
    # compute te area of the bounding boxes and sort the bounding boxes by
    # the score confidence of the bounding box
    area = (x2-x1)*(y2-y1)
    cl = [x for x in range(1,FLAGS.num_classes)]
    #scores_per_class = np.hsplit(scores, cl)
    #print("scores_per_class: {}\n{}\n{}".format(len(scores_per_class), scores_per_class[0].shape, scores_per_class[1].shape))
    #for score_class, c in zip(scores_per_class, cl):
    bbox_class = np.argmax(scores, axis=-1)
    scores_max = np.max(scores, axis=-1)
    order = list(scores_max.argsort()[::-1])

    # keep looping while ordered indexes still remain in the list
    while len(order):
        if len(picked_box) > max_outputs:
            break
        # pick the bbox with highest score and remove it from the list
        pick = order.pop()
        if scores_max[pick] < FLAGS.score_threshold:
            continue
        picked_box.append(pick)
        picked_class.append(bbox_class[pick])
        # find the largest (x,y) coordinates for the start of the bounding box
        # and the smallest (x,y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[pick], x1[order[0:]])
        yy1 = np.maximum(y1[pick], y1[order[0:]])
        xx2 = np.minimum(x2[pick], x2[order[0:]])
        yy2 = np.minimum(y2[pick], y2[order[0:]])
        # Compute the width, height and intersection of the bounding box
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w*h
        union = area[pick] + area[order[0:]] - intersection
    
        # Compute the ratio of overlap
        overlap = intersection / union
    
        # Eliminate the boxes with overlap higher than threshold
        idxs = overlap <= FLAGS.iou_threshold
        order = list(np.array(order)[idxs])

    return picked_box, picked_class

def draw_objects(draw, objs, labels, scale=(1.,1.), color_map={0:'red', 1:'green'}):
  """Draws the bounding box and label for each object."""
  x, y = scale[0]*416, scale[1]*416
  #print("x,y: \n\t{}, {}\n".format(x,y))
  for obj in objs:
    bbox = obj.bbox
    #print("bbox: ({},{},{},{})".format(bbox.xmin*x,bbox.ymin*y,bbox.xmax*x,bbox.ymax*y))
    draw.rectangle([(bbox.xmin*x, bbox.ymin*y),
                    (bbox.xmax*x, bbox.ymax*y)],
                   outline=color_map[obj.id])
    draw.text((bbox.xmin*x - 10, bbox.ymin*y - 10),
              '%s\n%.3f' % (labels.get(obj.id, obj.id), obj.score),
              fill=color_map[obj.id])

