#!/bin/python
import glob
import imageio
import random
import scipy
import numpy as np
import tensorflow as tf
from tf.keras import layers

def modify_picture(image):
    # flip
    if np.random.rand() > 0.40:
        image = np.fliplr(image)

    # rotate
    if np.random.rand() > 0.20:
        max_angle = 10
        angle = random.uniform(-max_angle, max_angle) * 2
        if angle > max_angle:
            angle = max_angle
        elif angle < -max_angle:
            angle = -max_angle
        image = scipy.misc.imrotate(image, angle)

    # shift
    if np.random.rand() > 0.20:
        max_offset = 22
        offset = random.uniform(-max_offset, max_offset) * 2
        if offset > max_offset:
            offset = max_offset
        elif offset < -max_offset:
            offset = -max_offset

        image = scipy.ndimage.shift(image, (int(offset), int(offset), 0), order=0)

    return image


def get_batches(batch_size):
    # Grab image and label paths
    img_src = '/tmp/train/*.png'
    image_paths = glob.glob(img_src)
    num_images = len(image_paths)
    # Shuffle training data
    random.shuffle(image_paths)
    #print('random.shuffle')
    print('number of images', num_images)

    # Loop through batches and grab images, yielding each batch
    #for batch_i in range(0, num_images, batch_size):
    num_batch = num_images // batch_size
    if num_batch < 12:
        num_batch = 12

    for batch_i in range(0, num_images, num_images//num_batch):
        #print('total images: ', len(image_paths), 'batch_start: ', batch_i)
        images = []
        labels = []

        # we need this if training set is too small
        if (batch_i + batch_size >= num_images):
            batch_i %= batch_size

        for image_file in image_paths[batch_i:batch_i + batch_size]:
            image = scipy.misc.imread(image_file)
            image = modify_picture(image)
            # RGB to BGR
            image = image[..., ::-1]
            image = image[200:800, 284:1084, :]

            images.append(image)

            if '_0' in image_file:
                labels.append(0)
            else:
                labels.append(1)

        yield np.array(images), np.array(labels)


def get_valid_batch():
    # Grab image and label paths
    #img_src = 'D:\\test_images\\*.png'
    img_src = '/tmp/train/*.png'
    image_paths = glob.glob(img_src)

    images = []
    labels = []
    for image_file in image_paths:
        #print('load ', image_file)
        image = scipy.misc.imread(image_file)
        # RGB to BGR
        image = image[..., ::-1]
        image = image[200:800, 284:1084, :]

        images.append(image)

        if '_0' in image_file:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(images), np.array(labels)


def tf_classifier(input, num_classes, is_training):
    input = (input - 128.) / 128

    # try tf.layers.separable_conv2d()
    for i in range(4):
        input = tf.layers.conv2d(input, filters=6*(i+1), kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)

    # print(input)
    input = tf.layers.conv2d(input, filters=6*(i+1), kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
    out_a = tf.layers.conv2d(input, filters=6*(i), kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    print(out_a)

    # from a
    out_b = tf.layers.conv2d(input, filters=6*(i+2), kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
    out_b = tf.layers.conv2d(out_b, filters=6*(i+1), kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    print(out_b)

    out_a = tf.layers.flatten(out_a)
    out_b = tf.layers.flatten(out_b)
    output = tf.concat([out_a, out_b], 1)
    print(output)

    output = tf.layers.dropout(output, rate=0.55, training=is_training)
    output = tf.layers.dense(output, 16, activation=tf.nn.relu)

    logits = tf.layers.dense(output, num_classes, name='tl_logits')
    return logits


def train_net(logits, num_classes):
    epochs = 60
    batch_size = 32
    saver = tf.train.Saver()

    with tf.Session() as sess:
        label = tf.placeholder(tf.int32, (None))
        label_one_hot = tf.one_hot(label, num_classes)
        learning_rate = tf.placeholder(tf.float32)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        valid_features, valid_labels = get_valid_batch()

        best_accuracy = 0
        best_loss = 100
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epochs):
            batch_idx = 0
            batches = get_batches(batch_size)
            for batch_features, batch_labels in batches:
                train_feed_dict = {
                    input: batch_features,
                    label: batch_labels,
                    is_training: True,
                    learning_rate: 0.001}
                _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=train_feed_dict)
                print("trainning epoch {}, batch {}, loss: {}".format(epoch_i, batch_idx, loss))
                batch_idx += 1

            valid_feed_dict = {
                input: valid_features,
                label: valid_labels,
                is_training: False}
            valid_loss, valid_accuracy = sess.run([cross_entropy_loss, accuracy_operation], feed_dict=valid_feed_dict)
            print("valid_loss epoch {}, loss: {}, accuracy: {}".format(epoch_i, valid_loss, valid_accuracy))
            if valid_accuracy >= .90:
                if (valid_accuracy > best_accuracy - 1e-4) and (valid_loss < best_loss + 1e-4):
                    saver.save(sess, './ckp_tf_classifier')
                    print("Model saved")
                    best_accuracy = valid_accuracy
                    best_loss = valid_loss

        #saver.save(sess, './ckp_tf_classifier')
        #print("Model saved")


if __name__ == '__main__':
    num_classes = 2
    image_shape = (600, 800, 3)

    input = tf.keras.layers.Input(shape=image_shape, name='image_input')
    print(input)

    is_training = tf.placeholder(tf.bool, name='is_training')
    logits = tf_classifier(input, num_classes, is_training)
    print(logits)
    train_net(logits, num_classes)
