import time
import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm

flags.DEFINE_string('data_dir', './my_data/voc2020_raw/VOCdevkit/VOC2020/',
                    'path to raw PASCAL VOC dataset')
flags.DEFINE_enum('split', 'train', [
                  'train', 'val'], 'specify train or val spit')
flags.DEFINE_string('output_file', './my_data/voc2020_train.tfrecord', 'output dataset')
flags.DEFINE_string('classes', './my_data/classes.names', 'classes file')
flags.DEFINE_string('emphasized_class', 'smoke', 'Select in which class to emphasize the dataset. \nOne of \'smoke\', \'fire\', \'all\' ')
flags.DEFINE_string('excluded_name', 'captures', 'define a name that it`s included in _train|_val .txt file in order to exclude it from dataset.')

def build_example(annotation, class_map):
    img_path = os.path.join(
        FLAGS.data_dir, 'JPEGImages', annotation['filename'])
    with open(img_path, 'rb') as img:
        img_raw = img.read()
        key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult_obj = []
    if 'object' in annotation:
        for obj in annotation['object']:
            if FLAGS.excluded_name in annotation['filename']:
                continue
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def main(_argv):
    with open(FLAGS.classes) as f:
        class_map = {name: idx for idx, name in enumerate(
            f.read().splitlines())}
    logging.info("Class mapping loaded: {}".format(class_map))

    writer = tf.io.TFRecordWriter(FLAGS.output_file)
    with open(os.path.join(FLAGS.data_dir, 'ImageSets', 'Main', '%s_%s.txt' % (FLAGS.emphasized_class, FLAGS.split))) as f:
        image_list = f.read().splitlines()
    logging.info("Image list loaded: %d", len(image_list))
    for image in tqdm.tqdm(image_list):
        name, _ = image.split()
        annotation_xml = os.path.join(
            FLAGS.data_dir, 'Annotations', name + '.xml')
        with open(annotation_xml) as f:
            annotation_xml = lxml.etree.fromstring(f.read())
        annotation = parse_xml(annotation_xml)['annotation']
        tf_example = build_example(annotation, class_map)
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info("Done") 


if __name__ == '__main__':
    app.run(main)
