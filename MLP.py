from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

dataset_path = '/Users/ahmedtawfik/PycharmProjects/OCR-for-Arabic-Letters/DBAHCL/Trial/Training'  # the dataset file
batch_size = 128
test_path = '/Users/ahmedtawfik/PycharmProjects/OCR-for-Arabic-Letters/DBAHCL/Trial/Testing'
test_size = 100

# Parameters
learning_rate = 0.1
num_steps = 500
display_step = 100

n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784*3 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

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

def OneHot(labels):
    out = np.zeros((batch_size, num_classes))
    for i in range(batch_size):
        out[i][labels[i]] = 1
    return out


filenames, labels = read_images(dataset_path)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(num_steps):
        sess.run(iterator.initializer)
        batch_x, batch_y = sess.run(next_element)
        batch_x.shape = (batch_size, num_input)
        batch_y = OneHot(batch_y)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    filenames, labels = read_images(test_path)
    testset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    testset = testset.map(_parse_function)
    testset = testset.shuffle(buffer_size=1000)
    testset = testset.batch(test_size)
    t_iterator = testset.make_initializable_iterator()
    test_element = t_iterator.get_next()
    sess.run(t_iterator.initializer)
    batch_x, batch_y = sess.run(test_element)
    batch_x.shape = (test_size, num_input)
    p_y = sess.run(logits, feed_dict={X: batch_x})
    p_y = sess.run(tf.argmax(p_y, 1))
    confusion_matrix = np.zeros((10, 10))
    correct = 0
    for i in range(test_size):
        confusion_matrix[batch_y[i]][p_y[i]] += 1
        if batch_y[i] == p_y[i]:
            correct += 1

    print(confusion_matrix)
    print(correct)