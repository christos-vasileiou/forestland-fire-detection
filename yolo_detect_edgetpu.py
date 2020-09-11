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
from yolo_edgetpu_lib import *

flags.DEFINE_integer('num_classes', 2, 'define the number of classes are going to be detected')
flags.DEFINE_string('classes', './fire_data/classes_edgetpu.names', 'path to classes file')
flags.DEFINE_string('model_path', 'edgetpu-segments/yolov3_tiny_firedetect_relu_edgetpu_edgetpu.tflite',
                    'path to weights file')
flags.DEFINE_string('image', './images/fire_91.jpg', 'path to input image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')


EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(model_file, edge_tpu=True):
    model_file, *device = model_file.split('@')
    if edge_tpu:
        return tflite.Interpreter(
          model_path=model_file,
          experimental_delegates=[
              tflite.load_delegate(EDGETPU_SHARED_LIB,
                                  {'device': device[0]} if device else {})
          ])
    else:
        return tflite.Interpreter(model_file)


def main(_argv):

    interpreter = make_interpreter(FLAGS.model_path, edge_tpu=True)
    interpreter.allocate_tensors()

    labels = load_labels(FLAGS.classes)
    image = Image.open(FLAGS.image)
    scaleW, scaleH = set_input(interpreter, image)

    print("resize image ratio: {}, {}\n".format(scaleW, scaleH))
    print('----INFERENCE TIME----')
    print('Note: The first inference is slow because it includes',
        'loading the model into Edge TPU memory.\n')

    for _ in range(3):
        print("\nInterpreter Invoke")
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs, inf_time = get_output(interpreter, (scaleW, scaleH))
        for t in inf_time:
            inference_time += t
        print('%.3f ms' % (inference_time*1000))

    print('-------RESULTS--------')
    if not objs:
        print('No objects detected')
    print(labels)
    for obj in objs:
        print(labels.get(obj.id, obj.id))
        print('  id:    ', obj.id)
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)

    image = image.convert('RGB')
    print("image dimension: {}".format(image.size))
    draw_objects(ImageDraw.Draw(image), objs, labels, (scaleW, scaleH))
    image.save('./tpu_output.jpg')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
