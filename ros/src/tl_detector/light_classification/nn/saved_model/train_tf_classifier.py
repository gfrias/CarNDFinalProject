
import glob
import imageio
import random
import scipy
import numpy as np
import tensorflow as tf


def modify_picture(image):
    # flip
    if np.random.rand() > 0.60:
        image = np.fliplr(image)

    # rotate
    if np.random.rand() > 0.60:
        max_angle = 5
        image = scipy.misc.imrotate(image, random.uniform(-max_angle, max_angle))

    # shift
    if np.random.rand() > 0.60:
        offset = random.uniform(-10, 10)
        image = scipy.ndimage.shift(image, (int(offset), int(offset), 0), order=0)

    return image


def get_batches(batch_size):
    # Grab image and label paths
    img_src = 'D:\\image\\*\\*.png'
    image_paths = glob.glob(img_src)
    # Shuffle training data
    random.shuffle(image_paths)
    #print('random.shuffle')

    # Loop through batches and grab images, yielding each batch
    for batch_i in range(0, len(image_paths), batch_size):
        #print('total images: ', len(image_paths), 'batch_start: ', batch_i)
        images = []
        labels = []
        for image_file in image_paths[batch_i:batch_i + batch_size]:
            image = scipy.misc.imread(image_file)
            image = modify_picture(image)
            # RGB to BGR
            image = image[..., ::-1]

            images.append(image)

            if '_0' in image_file:
                labels.append(0)
            elif '_1' in image_file:
                labels.append(1)
            else:
                labels.append(2)

        yield np.array(images), np.array(labels)


def get_valid_batch():
    # Grab image and label paths
    img_src = 'D:\\test_images\\*.png'
    image_paths = glob.glob(img_src)

    images = []
    labels = []
    for image_file in image_paths:
        image = scipy.misc.imread(image_file)
        image = modify_picture(image)
        # RGB to BGR
        image = image[..., ::-1]

    images.append(image)

    if '_0' in image_file:
        labels.append(0)
    elif '_1' in image_file:
        labels.append(1)
    else:
        labels.append(2)

    return np.array(images), np.array(labels)

def tf_classifier(input, num_classes, is_training):
    input = (input - 128.) / 128

    for i in range(4):
        input = tf.layers.conv2d(input, filters=6*(i+1), kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)

    # print(input)
    input = tf.layers.conv2d(input, filters=6*(i+1), kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
    out_a = tf.layers.conv2d(input, filters=6*(i+1), kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    print(out_a)

    # from a
    out_b = tf.layers.conv2d(out_a, filters=6*(i+3), kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
    out_b = tf.layers.conv2d(out_b, filters=6*(i+3), kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    print(out_b)

    out_a = tf.layers.flatten(out_a)
    out_b = tf.layers.flatten(out_b)
    output = tf.concat([out_a, out_b], 1)
    print(output)

    output = tf.layers.dropout(output, rate=0.5, training=is_training)
    output = tf.layers.dense(output, 100, activation=tf.nn.relu)

    logits = tf.layers.dense(output, num_classes, name='tl_logits')
    return logits


def train_net(logits, num_classes):
    epochs = 50
    batch_size = 64
    saver = tf.train.Saver()

    with tf.Session() as sess:
        label = tf.placeholder(tf.int32, (None))
        label_one_hot = tf.one_hot(label, num_classes)
        learning_rate = tf.placeholder(tf.float32)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

        valid_features, valid_labels = get_valid_batch()
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
            valid_loss = sess.run([cross_entropy_loss], feed_dict=valid_feed_dict)
            print("valid_loss epoch {}, loss: {}".format(epoch_i, valid_loss))

        saver.save(sess, './ckp_tf_classifier')
        print("Model saved")


if __name__ == '__main__':
    num_classes = 3
    image_shape = (600, 800, 3)

    input = tf.layers.Input(shape=image_shape, name='image_input')
    print(input)

    is_training = tf.placeholder(tf.bool, name='is_training')
    logits = tf_classifier(input, num_classes, is_training)
    print(logits)
    train_net(logits, num_classes)
