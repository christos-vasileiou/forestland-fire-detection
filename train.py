from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
import cv2
import os
import datetime
from tensorflow.keras.mixed_precision import experimental as tf_mixed_precision
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_tiny_anchors, 
    yolo_anchor_masks, yolo_tiny_anchor_masks,
    adjust_yolo_anchors, 
    experimentalYoloPrecision, experimentalYoloRecall
)
from yolov3_tf2.utils import (
    freeze_all, print_all_layers, setup_accelerator
)
import yolov3_tf2.dataset as dataset
import datetime

flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_string('mixed_precision', 'float32', 'model to be trained is quantized or not')
flags.DEFINE_string('pretrained_mixed_precision', 'float32', 'pretrained weights are quantized or not. if not, they will be transformed considering the value of `quantized` ')
flags.DEFINE_boolean('pdetails', False, 'Print the whole model Summary() and plot it to .png')
flags.DEFINE_string('output_weights', './checkpoints/yolov3_trained.tf',
                    'path to store trained weights file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'no_darknet', 'no_output', 'frozen', 'fine_tune', 'all'],
                  'none: Training from scratch, '
                  'no_darknet: Transfer all but darknet.'
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only'
                  'all: Transfer all layers, without freezing')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_float('ignore_thresh', 0.7, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('pretrained_weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_string('accelerator', 'none', 'Training using accelerator: [`none` | `gpu` | `tpu`] ')
flags.DEFINE_boolean('metrics', False, 'True: fit model with `mAP` metric ')
flags.DEFINE_string('log_output_fit', 'logs', 'specify the path to display the fit`s history'  )
flags.DEFINE_boolean('binary_class_loss', False, 'If True compute classes loss by using binary_crossentropy, else by using sparse_categorical_crossentropy')

DEFAULT = '\033[00m'
GREEN = '\033[92m'

def main(_argv):
    # Setting up the accelerator
    strategy = setup_accelerator(FLAGS.accelerator)

    if 'TPU' in FLAGS.accelerator:
        with strategy.scope():
            if FLAGS.tiny:
                model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes, mixed_precision=FLAGS.mixed_precision)
                anchors = adjust_yolo_anchors(yolo_tiny_anchors, FLAGS.size)
                anchor_masks = yolo_tiny_anchor_masks
            else:
                model = YoloV3(FLAGS.size, training=True, 
                            classes=FLAGS.num_classes, mixed_precision=FLAGS.mixed_precision)
                anchors = adjust_yolo_anchors(yolo_anchors, FLAGS.size)
                anchor_masks = yolo_anchor_masks
    else:
        if FLAGS.tiny:
            model = YoloV3Tiny(FLAGS.size, training=True,
                        classes=FLAGS.num_classes, mixed_precision=FLAGS.mixed_precision)
            anchors = adjust_yolo_anchors(yolo_tiny_anchors, FLAGS.size)
            anchor_masks = yolo_tiny_anchor_masks
        else:
            model = YoloV3(FLAGS.size, training=True, 
                        classes=FLAGS.num_classes, mixed_precision=FLAGS.mixed_precision)
            anchors = adjust_yolo_anchors(yolo_anchors, FLAGS.size)
            anchor_masks = yolo_anchor_masks

    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    train_dataset = train_dataset.shuffle(buffer_size=512).batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    
    # Configure the model for transfer learning
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    elif FLAGS.transfer in ['no_darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes

        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.pretrained_weights_num_classes or FLAGS.num_classes,
                mixed_precision=FLAGS.pretrained_mixed_precision
            )
        else:
            model_pretrained = YoloV3(
                    FLAGS.size, training=True, classes=FLAGS.pretrained_weights_num_classes or FLAGS.num_classes,
                    mixed_precision=FLAGS.pretrained_mixed_precision
            )
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == 'no_darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))
        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(l.name).get_weights())
                    freeze_all(l)
    else:
        # All other transfer require matching classes
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'all':
            # fine_tune the whole model.
            for i, l in enumerate(model.layers):
                print(i, "layer: ", l.name)
                freeze_all(l, frozen=False)
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes, ignore_thresh=FLAGS.ignore_thresh, dtype=FLAGS.mixed_precision)
            for mask in anchor_masks]
    
    logging.info("\nloss: {}".format(loss))
    for l in loss:
        print(l)
    logging.info("global policy: {}".format(tf_mixed_precision.global_policy()))
    logging.info("global policy loss: {}\n".format(tf_mixed_precision.global_policy().loss_scale))
    # Print model summary and plot it to .png 
    print_all_layers(model, p_details=FLAGS.pdetails)
    if FLAGS.tiny:
        tf.keras.utils.plot_model(model, to_file='yolov3_tiny.png', show_shapes=True, show_layer_names=False)
    else:
        tf.keras.utils.plot_model(model, to_file='yolov3.png', show_shapes=True, show_layer_names=False)

    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training

        #optimizer = tf_mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        y = []
        y_val = []
        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info("{}_val_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)

            logging.info(GREEN+"{}, train: {}, val: {}\33".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy())+DEFAULT)

            y.append(avg_loss.result().numpy())
            y_val.append(avg_val_loss.result().numpy())

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            if epoch%4 == 0:
                save_epoch = epoch
                model.save_weights(
                    'checkpoints/yolov3_train_{}.tf'.format(save_epoch))
                logging.info(GREEN+'checkpoints/yolov3_train_{}.tf weights saved'.format(save_epoch)+DEFAULT)
        
    else:
        metrics = None
        if FLAGS.metrics:
            thresholds = [x for x in np.linspace(0,1,11)]
            precision = []
            #recall = []
            for mask in anchor_masks:
                precision.append([experimentalYoloPrecision(anchors=anchors[mask], num_classes=FLAGS.num_classes, thresholds=thresholds, name='precision')])
                #recall.append([experimentalYoloRecall(anchors=anchors[mask], num_classes=FLAGS.num_classes, thresholds=thresholds, name='recall')])
            #recall = [experimentalYoloRecall(anchors=anchors[mask], num_classes=FLAGS.num_classes, thresholds=thresholds, name='recall') for mask in anchor_masks]
            metrics = []
            for p in zip(precision):
                metrics.append(p)

        if 'TPU' in FLAGS.accelerator:
            with strategy.scope():
                model.compile(optimizer=optimizer, loss=loss,
                              metrics = metrics, run_eagerly=(FLAGS.mode=='eager_fit'))
        else:
            model.compile(optimizer=optimizer, loss=loss,
                          metrics = metrics, run_eagerly=(FLAGS.mode=='eager_fit'))
        
        callbacks = [
            ModelCheckpoint(FLAGS.output_weights, verbose=1, save_best_only=True, save_weights_only=True),
            TensorBoard(log_dir=FLAGS.log_output_fit+'-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)
        ]
        
        print("\n--- START FITTING ---\n")
        training_history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset,
                            workers=2)

        logging.info(" > Average loss: \t{}".format(np.average(training_history.history['loss'])))
        logging.info(" > Average val loss: \t{}".format(np.average(training_history.history['val_loss'])))
        if FLAGS.metrics:
            for i, k in enumerate(list(training_history.history.keys())):
                if 'precision' in k:
                    logging.info(" > {0}, mAP={1:.4}%".format(k, np.average(training_history.history[k])*100))
                if 'recall' in k:
                    logging.info(" > {0}, mAR={1:.4}%".format(k, np.average(training_history.history[k])*100))
                
        #logging.info(" > History dictionary: \t{}".format(training_history.history))
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
