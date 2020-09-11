# Export anchors with K-means algorithm for better fitting in the fire-smoke dataset.

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import lxml.etree
import tqdm
import glob
import os
from absl import app, flags, logging
from absl.flags import FLAGS
from sklearn.cluster import KMeans
from IPython.display import Image, display

#%matplotlib inline
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('data_dir', './my_data/fire_smoke_raw/FIRESMOKEdevkit/FIRESMOKE2020/',
                    'path to fire-smoke dataset')
flags.DEFINE_string('classes', './my_data/classes.names', 'path to class file')
flags.DEFINE_string('output', './my_data/anchors.txt', 'anchors` file to be used for training')

def read_classes():
    with open(FLAGS.classes) as f:
        class_map = {name: idx for idx, name in enumerate(f.read().splitlines()) }
    logging.info("{} classes read!".format(len(class_map)))
    return class_map

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


def read_anchors():
    anchors = []
    width, height = 0, 0
    i=0
    for name in tqdm.tqdm(os.listdir(FLAGS.data_dir+'Annotations')):
        annotation_xml = os.path.join(FLAGS.data_dir, 'Annotations', name)
        with open(annotation_xml) as f:
            try:
                annotation_xml = lxml.etree.fromstring(f.read())
            except:
                print("Problem parsing: \n{}\n".format(name))
        annotation = parse_xml(annotation_xml)['annotation']
        if ('captures' not in annotation['filename']):
            w = annotation['size']['width']
            h = annotation['size']['height']
            for obj in annotation['object']:
                #print(obj['bndbox']['xmin'], '\n', w)
                xmin = float(obj['bndbox']['xmin']) / float(w)
                xmax = float(obj['bndbox']['xmax']) / float(w)
                ymin = float(obj['bndbox']['ymin']) / float(h)
                ymax = float(obj['bndbox']['ymax']) / float(h)
                anchors.append({'name': obj['name'], 'bndbox': [xmax-xmin, ymax-ymin] })

            width += int(w)
            height += int(h)
        i+=1

    return anchors, [width/i, height/i]

def k_means(anchors, class_map):
    if FLAGS.tiny:
        K = 6
    else:
        K = 9
    X = []
    Y = []
    for anchor in anchors:
        anch = np.array(anchor['bndbox'])
        cl = np.array([class_map[anchor['name']]])
        X.append(anch)
        Y.append(cl)

    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(Y.shape)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)    
    print("cluster's centers of anchors: \n{}".format(kmeans.cluster_centers_))
    
    return kmeans.cluster_centers_, X, Y

def is_colab():
    #TODO: check if run in other notebooks except colab.
    import sys
    COLAB = 'google.colab' in sys.modules
    return COLAB

def plot(centers, X, name='dataset-variance.jpg'):
    fig = plt.figure(figsize=(6,6))    
    print("X:\n{}".format(X))
    print("centers:\n{}\n".format(centers))
    plt.scatter(X[...,0], X[...,1], color='k')
    color_map = {0: 'r', 1: 'r', 2: 'r', 3: 'r', 4: 'r', 5: 'r', 6: 'r', 7: 'r', 8: 'r'}
    marker_map = {0: '^', 1: '^', 2: '^', 3: 'o', 4: 'o', 5: 'o', 6: 's', 7: 's', 8: 's'}
    for i, center in enumerate(zip(color_map, centers)):
        plt.scatter(center[1][0], center[1][1], color=color_map[i], marker=marker_map[i])
    fig.savefig(name)

def sort_by_area(anchors, size):
    area = {}
    anch = {}
    sorted_anch = []
    for i, anchor in enumerate(anchors):
        area[str(i)] = anchor[0]*anchor[1]
        anch[str(i)] = np.array(anchor)
    area = sorted(area.items(), key=lambda kv: kv[1])
    for i in area:
        anch[i[0]] *= size
        sorted_anch.append(anch[i[0]])
    sorted_anch = np.round(np.array(sorted_anch)).astype(int)
    print("sorted anchors: \n{}".format(sorted_anch))

    return sorted_anch

def export(centers):
    with open(FLAGS.output, 'wb') as f:
        np.savetxt(f, centers, fmt='%d')
    
def main(_argv):
    print('dataset: ')
    print('\t{}'.format(FLAGS.data_dir), '\n'*2)
    class_map = read_classes()
    anchors, size = read_anchors()
    cluster_centers, X_data, Y_data = k_means(anchors, class_map)
    plot(cluster_centers*size, X_data*size, name='dataset-variance.jpg')
    cluster_centers = sort_by_area(cluster_centers, size)
    plot(cluster_centers*size, X_data*size, name='dataset-variance-2.jpg')
    export(cluster_centers)
    if is_colab():
        display(Image('dataset-variance.jpg'))
    else:
        plt.show()

    print('\n'*2)

if __name__ == '__main__':
    app.run(main)