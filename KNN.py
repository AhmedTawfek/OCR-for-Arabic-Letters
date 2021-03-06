import numpy as np
import tensorflow as tf
import os


# tf Graph Input
max_value = tf.placeholder(tf.int64, shape=[])
size = tf.placeholder(tf.int64)
num_classes = 10
num_input = 2352
xtr = tf.placeholder("float", [None, num_input])
xte = tf.placeholder("float", [num_input])

dataset_path = '/Users/ahmedtawfik/PycharmProjects/OCR-for-Arabic-Letters/DBAHCL/Trial/Training'  # the dataset file or root folder path.
batch_size = 900
test_size = 50

def read_images(dataset_path):
    imagepaths, labels = list(), list()
    # An ID will be affected to each sub-folders by alphabetical order
    label = 0
    classes = sorted(os.walk(dataset_path).__next__()[1])
    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.walk(c_dir).__next__()
        # Add each image to the training set
        for sample in walk[2]:
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
    #training
    sess.run(iterator.initializer, feed_dict={size: batch_size})
    Xtr, Ytr = sess.run(next_element, feed_dict={size: batch_size})
    Ytr = OneHot(Ytr, batch_size)
    Xtr.shape = (batch_size, num_input)
    Ytr.shape = (batch_size, num_classes)
    #testing
    sess.run(iterator.initializer, feed_dict={size: test_size})
    Xte, Yte = sess.run(next_element, feed_dict={size: test_size})
    Yte = OneHot(Yte, test_size)
    Xte.shape = (test_size, num_input)
    Yte.shape = (test_size, num_classes)
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
            accuracy += 1. / len(Xte)
    print(ConfusionMatrix)
    print("Done!")
    print("Accuracy:", accuracy)
