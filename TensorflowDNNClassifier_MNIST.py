# -*- coding: utf-8 -*-
"""
This code is from a tutorial I found online for using DNNClassifier on the MNIST dataset.
The code itself in the tutorial has some outdated tensorflow code. I have fixed these and
made notes in the code below. The main power in this code is it takes in array input, which
is how image data is formatted, and has very few lines of code.

Link to the tutorial: https://codeburst.io/use-tensorflow-dnnclassifier-estimator-to-classify-mnist-dataset-a7222bf9f940 
"""

import tensorflow as tf
import numpy as np

#this is a new way to read the dataset, the tutorial uses: input_data.read_data_sets()
mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')
# can use below to explore the dataset
# print(mnist)
# dataset_mnist = mnist[0][0]
# labels_mnist = mnist[0][1]
# print(dataset_mnist)
# print(labels_mnist)
# print(dataset_mnist.shape)

#define one feature with shape of each array 28x28
feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]
print(feature_columns)

#this is updated from the tutorial because of how keras mnist is formatted
def input(dataset):
    return dataset[0], dataset[1].astype(np.int32)

# Build 2 layer DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    #these two hidden layer sizes are between the input and output layer sizes of 784 and 10
    hidden_units=[256, 32],
    #this is updated from the tutorial, it's how the documenation says to enter optimizers
    optimizer = 'Adam',
    n_classes=10,
    dropout=0.1
)

# Create training input function
# this is updated from the tutorial, was tf.estimator.inputs...
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    #fixed the below from tutorial, keras mnist if formatted differently
    x={"x": input(mnist[0])[0]},
    y=input(mnist[0])[1],
    num_epochs=None,
    batch_size=50,
    shuffle=True
)

# Train the model, note there are 6000 samples in mnist dataset
classifier.train(input_fn=train_input_fn, steps=100000)

# Create testing input function
# this is updated from the tutorial, was tf.estimator.inputs...
test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    #fixed the below from tutorial, keras mnist if formatted differently
    x={"x": input(mnist[1])[0]},
    y=input(mnist[1])[1],
    num_epochs=1,
    shuffle=False
)

# Test the model
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))

#%%
"""Extra code"""
# this input function is possible to use, however data has to be flattened into 1D array
# def input_fn_using_flatten_data(features, labels, training=True, batch_size=64): #was 256
#     """An input function for training or evaluating"""
#     # Convert the inputs to a Dataset.
#     dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

#     # Shuffle and repeat if you are in training mode.
#     if training:
#         dataset = dataset.shuffle(150).repeat() # dataset.shuffle(buffer_size, seed=seed, reshuffle_each_iteration=True).repeat(count)
#     return dataset.batch(batch_size)

# classifier.train(input_fn=lambda: input_fn_using_flatten_data(dataset_mnist, labels_mnist), steps=100000)