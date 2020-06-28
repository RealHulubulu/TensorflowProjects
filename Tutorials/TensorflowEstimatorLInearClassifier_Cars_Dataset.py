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

import pandas as pd
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load dataset
dfAll = pd.read_csv('C:/Users/karas/jupyter/car.txt', names=["Buying", "Maint", "Doors", "Persons", 
                                                             "Lug_Boot", "Safety", "Classif"])

# dfAll.describe()

#these lists are for keeping the accuracy with and without crossed columns
base_accuracy_list = []
modified_accuracy_list = []

# this has been moved outside of the below for loop to better serve as test set
train_and_valid, test = train_test_split(dfAll, test_size=0.05) 
y_test = test.pop("Classif")

# Each run creates two new models, one with and one without crossed columns, saves accuracy each time
for acc_counter in range(1):
    
    train, validate = train_test_split(train_and_valid, test_size=0.2)
    # train, test = train_test_split(train, test_size=0.05)    

    Y_COL = "Classif"
    #doing pop below removes Classif col from train, test, validate
    y_train = train.pop(Y_COL)
    # y_test = test.pop(Y_COL)
    validate_y = validate.pop(Y_COL)

    # creating feature columns 
    CATEGORICAL_COLUMNS = ["Buying", "Maint", "Lug_Boot", "Safety", "Doors", "Persons"]
    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS: # can also use train.keys()
      vocabulary = train[feature_name].unique()
      # print(vocabulary)
      feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    
    # creating y feature columns for using y labels for classification
    y_feature_columns = []
    y_vocab = ["unacc", "acc", "good", "vgood"]
    feature_name = Y_COL
    y_feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, y_vocab))
    # print(y_vocab)
    # print(y_feature_columns)
    
    def make_input_fn(data_df, label_df, num_epochs=50, shuffle=True, batch_size=32):
        """returns the bottom function for a reusable input function, format from tensorflow website"""
        def input_function():
            """standard input function for models"""
            ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
            if shuffle:
                ds = ds.shuffle(1000)
            ds = ds.batch(batch_size).repeat(num_epochs)
            return ds
        return input_function
    
    train_input_fn = make_input_fn(train, y_train)
    validate_input_fn = make_input_fn(validate, validate_y, num_epochs=1, shuffle=False)

    ## this just shows samples of feature keys and batch class/labels
    # ds = make_input_fn(train, y_train, batch_size=10)()
    # for feature_batch, label_batch in ds.take(1):
    #   print('Some feature keys:', list(feature_batch.keys()))
    #   print()
    #   print('A batch of class:', feature_batch['Buying'].numpy())
    #   print()
    #   print('A batch of Labels:', label_batch.numpy())
     
    # Create model with basic feature columns
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=4, label_vocabulary=y_vocab)
    # Train model
    linear_est.train(train_input_fn)
    # Validate model
    result = linear_est.evaluate(validate_input_fn)
    clear_output()
    print(result)
    # print(result['accuracy'])
    base_accuracy_list.append(result['accuracy'])

    # create crossed columns for features that could relate, this is feature engineering
    #CATEGORICAL_COLUMNS = ["Buying", "Maint", "Lug_Boot", "Safety", "Doors", "Persons"]
    Buying_x_Maint = tf.feature_column.crossed_column(['Buying', 'Maint'], hash_bucket_size=10000)
    Buying_x_Safety = tf.feature_column.crossed_column(['Buying', 'Safety'], hash_bucket_size=10000)
    Lug_x_Maint = tf.feature_column.crossed_column(['Lug_Boot', 'Maint'], hash_bucket_size=10000)
    derived_feature_columns = [Buying_x_Maint, Buying_x_Safety, Lug_x_Maint]
    
    # Create model with crossed features columns
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns, n_classes=4, label_vocabulary=y_vocab)
    # Train model
    linear_est.train(train_input_fn)
    # Validate Model
    result = linear_est.evaluate(validate_input_fn)
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

# this is for testing against the test set, uses the crossed feature column model
y_vocab_labels = ["unacc", "acc", "good", "vgood"]
pred_accuracy = 0

def pred_input_fn(features, batch_size=256):
    """An input function for prediction. Convert the inputs to a Dataset without labels"""
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

prediction = linear_est.predict(input_fn=lambda: pred_input_fn(test))

## below creates a list of all prediction accuracies for a given index, here [1] == "acc"
## making the list affects output
# prediction_list = list(prediction)
# probs = pd.Series([pred['probabilities'][1] for pred in prediction_list])
# print(prediction_list)

print()
print("Testing using the improved modified model")
for pred_dict, expec in zip(prediction, y_test):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        y_vocab_labels[class_id], 100 * probability, expec))
    if y_vocab_labels[class_id] == expec:
        pred_accuracy += 1
print('Prediction accuracy of test set is {}%' .format(100*(pred_accuracy/len(y_test))))
