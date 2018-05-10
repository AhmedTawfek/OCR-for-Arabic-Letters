import numpy as np
import tensorflow as tf
import os

dataset_path = '/Users/hejazi/Downloads/DBAHCL/Trial/Training'  # the dataset file or root folder path.

# tf Graph Input
max_value = tf.placeholder(tf.int64, shape=[])
size = tf.placeholder(tf.int64)
num_classes = 10
num_input = 2352
xtr = tf.placeholder("float", [None, num_input])
xte = tf.placeholder("float", [num_input])

dataset_path = '/Users/hejazi/Downloads/DBAHCL/Trial/Training'  # the dataset file or root folder path.
batch_size = 900
fold_size = batch_size//10
test_size = 50


def read_images(dataset_path):
    imagepaths, labels = list(), list()
    # An ID will be affected to each sub-folders by alphabetical order
    label = 0
    # List the directory
    try:  # Python 2
        classes = sorted(os.walk(dataset_path).next()[1])
    except Exception:  # Python 3
        classes = sorted(os.walk(dataset_path).__next__()[1])
    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        try:  # Python 2
            walk = os.walk(c_dir).next()
        except Exception:  # Python 3
            walk = os.walk(c_dir).__next__()
        # Add each image to the training set
        for sample in walk[2]:
            # Only keeps jpeg images
            if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                imagepaths.append(os.path.join(c_dir, sample))
                labels.append(label)
        label += 1
    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    return imagepaths, labels


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 28, 28)
    return image_resized, label


def OneHot(labels, d_size):
    out = np.zeros((d_size, num_classes))
    for i in range(d_size):
        out[i][labels[i]] = 1
    return out


filenames, labels = read_images(dataset_path)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(size)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
pred = tf.argmin(distance, 0)
accuracy = 0.
init = tf.global_variables_initializer()
ConfusionMatrix = np.zeros((10, 10))

with tf.Session() as sess:
    sess.run(init)
    # training
    sess.run(iterator.initializer, feed_dict={size: batch_size})
    X, Y = sess.run(next_element, feed_dict={size: batch_size})
    Y = OneHot(Y, batch_size)
    X.shape = (batch_size, num_input)
    Y.shape = (batch_size, num_classes)
    for j in range(10):
        Xte = X[j * fold_size:j * fold_size + fold_size]
        Yte = Y[j * fold_size:j * fold_size + fold_size]
        Xtr = np.append(X[:j * fold_size], X[j * fold_size + fold_size:], axis=0)
        Ytr = np.append(Y[:j * fold_size], Y[j * fold_size + fold_size:], axis=0)
        #print(Ytr.shape)
        for i in range(len(Xte)):
            # Get nearest neighbor
            nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
            # Get nearest neighbor class label and compare it to its true label
            print
            "Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Yte[i])
            # Calculate accuracy
            ConfusionMatrix[np.argmax(Yte[i])][np.argmax(Ytr[nn_index])] += 1
            if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
                accuracy += 1. / batch_size
    print(ConfusionMatrix)
    print("Done!")
    print("Accuracy:", accuracy)
