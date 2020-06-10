#!/usr/bin/env python
# coding: utf-8

"""
This code is from Tensorflow tutorials. It has some modifications. I used the same estimator 
linear classifier on a car dataset different from the tutorial. I created a second output 
that shows a modified output using crossed columns in the learning. The original and modified
ouput accuracies are displayed at the end. The validation at the end uses the modified model.
You can set the number of models to create to compare their accuracy to each other.
Link to the tutorial: https://www.tensorflow.org/tutorials/estimator/linear
Link to the dataset: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

from sklearn.model_selection import train_test_split

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf




# Load dataset.
# dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
# dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
dfAll = pd.read_csv('C:/Users/karas/jupyter/car.txt', names=["Buying", "Maint", "Doors", "Persons", 
                                                             "Lug_Boot", "Safety", "Classif"])


# dfAll.describe()
base_accuracy_list = []
modified_accuracy_list = []

    
for acc_counter in range(1):

    train, test_no_valid = train_test_split(dfAll, test_size=0.2)
    test, validate = train_test_split(test_no_valid, test_size=0.05)    

    #doing pop below removes Classif col    
    y_train = train.pop("Classif")
    y_test = test.pop("Classif")
    
    validate_y = validate.pop('Classif')

    
    CATEGORICAL_COLUMNS = ["Buying", "Maint", "Lug_Boot", "Safety", "Doors", "Persons"]
    # NUMERIC_COLUMNS = ["Persons"]
    Y_COL = ["Classif"]
    
    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
      vocabulary = train[feature_name].unique()
      # print(vocabulary)
      feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    
    # for feature_name in NUMERIC_COLUMNS:
    #   feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    
    
    y_feature_columns = []
    y_vocab = ["unacc", "acc", "good", "vgood"]
    feature_name = "Classif"
    y_feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, y_vocab))
    print(y_vocab)
    # print(y_feature_columns)
    # print(abcd)
    
    
    def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
      def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
          ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
      return input_function
    
    train_input_fn = make_input_fn(train, y_train)
    test_input_fn = make_input_fn(test, y_test, num_epochs=1, shuffle=False)

    
    ## this just shows samples of feature keys and batch class/labels
    # ds = make_input_fn(train, y_train, batch_size=10)()
    # for feature_batch, label_batch in ds.take(1):
    #   print('Some feature keys:', list(feature_batch.keys()))
    #   print()
    #   print('A batch of class:', feature_batch['Buying'].numpy())
    #   print()
    #   print('A batch of Labels:', label_batch.numpy())
     
    
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=4, label_vocabulary=y_vocab)
    linear_est.train(train_input_fn)
    result = linear_est.evaluate(test_input_fn)
    
    clear_output()
    print(result)
    # print(result['accuracy'])
    base_accuracy_list.append(result['accuracy'])

    #CATEGORICAL_COLUMNS = ["Buying", "Maint", "Lug_Boot", "Safety", "Doors", "Persons"]
    Buying_x_Maint = tf.feature_column.crossed_column(['Buying', 'Maint'], hash_bucket_size=10000)
    Buying_x_Safety = tf.feature_column.crossed_column(['Buying', 'Safety'], hash_bucket_size=10000)
    Lug_x_Maint = tf.feature_column.crossed_column(['Lug_Boot', 'Maint'], hash_bucket_size=10000)
    derived_feature_columns = [Buying_x_Maint, Buying_x_Safety, Lug_x_Maint]
    
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns, n_classes=4, label_vocabulary=y_vocab)
    linear_est.train(train_input_fn)
    result = linear_est.evaluate(test_input_fn)
    
    clear_output()
    print(result)
    # print(result['accuracy'])
    modified_accuracy_list.append(result['accuracy'])

print()
print("Original Accs: ", base_accuracy_list)
print("Modified Accs: ", modified_accuracy_list)    
base_avg_acc = sum(base_accuracy_list)/len(base_accuracy_list)
modified_avg_acc =sum(modified_accuracy_list)/len(modified_accuracy_list)
print()
print("Original avg acc: " + str(base_avg_acc))
print("Modified avg acc: " + str(modified_avg_acc))
print()
#%%
# this is for testing against the validation set
y_vocab_labels = ["unacc", "acc", "good", "vgood"]
pred_accuracy = 0

def pred_input_fn(features, batch_size=256):
    """An input function for prediction. Convert the inputs to a Dataset without labels"""
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

prediction = linear_est.predict(input_fn=lambda: pred_input_fn(validate))

## below creates a list of all prediction accuracies for a given index
## making the list affects output
# prediction_list = list(prediction)
# probs = pd.Series([pred['probabilities'][1] for pred in prediction_list])
# print(prediction_list)

print()
print("Validation using the improved modified model")
for pred_dict, expec in zip(prediction, validate_y):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    
    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        y_vocab_labels[class_id], 100 * probability, expec))
    
    if y_vocab_labels[class_id] == expec:
        pred_accuracy += 1
print()
print('Prediction accuracy of validation set is {}%' .format(100*(pred_accuracy/len(validate_y))))
