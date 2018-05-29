'''This .py file builds the neural network'''

# Importing the necessary packages
from imports import *

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 3], name='input')

if len(images_placeholder.shape) == 3:
    shape_value = np.array([-1, tf.Dimension(img_size), tf.Dimension(img_size), 1], dtype=np.int32)
    reshape_image_input = tf.reshape(images_placeholder, shape_value)
# Weights initialization
weights_list = []
weights_list.append(5)
weights_list.append(5)
weights_list.append(images_placeholder.shape[3])
weights_list.append(8)
weights_tensor = np.array([weights_list[0], weights_list[1], tf.Dimension(weights_list[2]), weights_list[3]], dtype = np.int32)
W_conv1 = tf.Variable(tf.truncated_normal(weights_tensor, stddev = 0.05),name='W_conv1')

# Bias initialization
b_conv1 = tf.Variable(tf.constant(0.05,shape=[8]),name='b_conv1')

# Building a convolutional layer using tf.nn.conv2d function
h_conv1 = tf.nn.conv2d(images_placeholder, W_conv1, strides=[1, 1, 1, 1], padding="VALID") + b_conv1

# Applying activation to the output of convolutional layer
h_conv1 = tf.nn.relu(h_conv1)

# Building max pooling using tf.nn.max_pool function
h_max2 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Weights initialization
weights_list = []
weights_list.append(7)
weights_list.append(7)
weights_list.append(h_max2.shape[3])
weights_list.append(8)
weights_tensor = np.array([weights_list[0], weights_list[1], tf.Dimension(weights_list[2]), weights_list[3]], dtype = np.int32)
W_conv3 = tf.Variable(tf.truncated_normal(weights_tensor, stddev = 0.05),name='W_conv3')

# Bias initialization
b_conv3 = tf.Variable(tf.constant(0.05,shape=[8]),name='b_conv3')

# Building a convolutional layer using tf.nn.conv2d function
h_conv3 = tf.nn.conv2d(h_max2, W_conv3, strides=[1, 1, 1, 1], padding="VALID") + b_conv3

# Applying activation to the output of convolutional layer
h_conv3 = tf.nn.relu(h_conv3)

# Building max pooling using tf.nn.max_pool function
h_max4 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Weights initialization
weights_list = []
weights_list.append(9)
weights_list.append(9)
weights_list.append(h_max4.shape[3])
weights_list.append(16)
weights_tensor = np.array([weights_list[0], weights_list[1], tf.Dimension(weights_list[2]), weights_list[3]], dtype = np.int32)
W_conv5 = tf.Variable(tf.truncated_normal(weights_tensor, stddev = 0.05),name='W_conv5')

# Bias initialization
b_conv5 = tf.Variable(tf.constant(0.05,shape=[16]),name='b_conv5')

# Building a convolutional layer using tf.nn.conv2d function
h_conv5 = tf.nn.conv2d(h_max4, W_conv5, strides=[1, 1, 1, 1], padding="VALID") + b_conv5

# Applying activation to the output of convolutional layer
h_conv5 = tf.nn.relu(h_conv5)

# Building max pooling using tf.nn.max_pool function
h_max6 = tf.nn.max_pool(h_conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

if len(h_max6.shape) == 4:
    # Flattening layer
    flat_shape = h_max6.shape[1] * h_max6.shape[2] * h_max6.shape[3]
    shape_value = np.array([-1, tf.Dimension(flat_shape)], dtype=np.int32)
    h_max6 = tf.reshape(h_max6, shape_value)

weight_shape = np.array([tf.Dimension(h_max6.shape.as_list()[1]), 10], dtype=np.int32)
# Let's define trainable weights and biases
W_fc_7 = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.05))
b_fc_7 = tf.Variable(tf.constant(0.05, shape=[10]))
# Fully-connected layer created using tf.matmul function
h_fc_7 = tf.matmul(h_max6, W_fc_7, name = 'y_pred') + b_fc_7

# y_pred contains the predicted probability of each class for each input image
y_pred = h_fc_7