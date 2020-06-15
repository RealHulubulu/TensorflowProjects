#!/usr/bin/env python
# coding: utf-8

"""
This code is from Tensorflow tutorials. It has some modifications. This script uses the 
Estimator DNNClassifier on standard CSV datasets. The DNNClassifier is typically used for
more complex datasets with many features and that are non-linear. The example below uses
a dataset from UC Irvine as an example. The accuracy changes drastically per run due to 
the limited features and linear nature of this dataset.

One thing that differs in DNNClassifier from the LinearClassifier is using steps instead of
epochs. You can either use epochs or steps as they relate by the equation: 
    steps = epochs * (samples / batch_size)
Each step passes through one batch_size worth of data. Using steps as the training parameter
instead of epochs means you know how many batches went through the model. Unless you set the
steps specifically, the model will pass only a partial epoch towards the end.

Link to the tutorial: https://www.tensorflow.org/tutorials/estimator/premade
Link to dataset: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
Link to UC Irvine database resource: https://archive.ics.uci.edu/ml/datasets.php
Link to info on epoch vs steps: https://stackoverflow.com/questions/38340311/what-is-the-difference-between-steps-and-epochs-in-tensorflow#:~:text=6%20Answers&text=An%20epoch%20usually%20means%20one,a%20much%20larger%20data%20set.
"""

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

#You can hard code in new datasets or create your own user input function: dfAll, label = data_input()
CSV_COLUMN_NAMES = ["Buying", "Maint", "Doors", "Persons", "Lug_Boot", "Safety", "Classif"] #car.names
dfAll = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=CSV_COLUMN_NAMES, header=0)
label = "Classif"


def split_data(data, test_size=0.2):
    train, test = train_test_split(data, test_size=test_size)
    return train, test
    
def data_prep(data, y_label_name):
    
    train, validate = split_data(data)
    train, test = split_data(train, test_size=0.05)
    
    # #doing pop below removes Classif col
    y_train = train.pop(y_label_name)
    y_test = test.pop(y_label_name) 
    validate_y = validate.pop(y_label_name)
    
    #y labels have to be strings to use y vocab
    y_train = y_train.astype(str)
    y_test = y_test.astype(str)
    validate_y = validate_y.astype(str)
    
    return train, test, validate, y_train, y_test, validate_y

def convert_labels_to_num(y_series):
    new_y = []
    y_vocab = list(y_series.unique())
    for item in y_series:
        if item == y_vocab[0]:
            new_y.append(0)
        elif item == y_vocab[1]:
            new_y.append(1)
        elif item == y_vocab[2]:
            new_y.append(2)
        else:
            new_y.append(3)
    new_y = pd.Series(new_y)
    return(new_y)

def cols_names(data, cat_or_num = False):
    if cat_or_num == True:
        list_cat = []
        list_num = []
        
        for col in data.columns:
            num = False
            cat = False
            for item in data[col]:
                if isinstance(item, int) or isinstance(item, float):
                    num = True
                else:
                    cat = True
            if num == True and cat == False:
                list_num.append(col)
            else:
                list_cat.append(col)
        return list_cat, list_num
        
    else:  
        list_col_names = []
        for col in data.columns:
            list_col_names.append(col)
        return list_col_names
    
def input_fn(features, labels, training=True, batch_size=64): #was 256
    """An input function for training or evaluating"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(150).repeat() # dataset.shuffle(buffer_size, seed=seed, reshuffle_each_iteration=True).repeat(count)
    return dataset.batch(batch_size)


def feature_col_creator(cat_col, num_col, train_data):
    feature_columns = []
    
    for feature_name in cat_col:
        vocabulary = train_data[feature_name].unique()
        # print(vocabulary)
        #This is how categorical columns work outside DNNs
        # feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
        # DNN does not take categorical directly, need to use indicator_column
        categorial_col = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
        feature_columns.append(tf.feature_column.indicator_column(categorial_col))
    
    for feature_name in num_col:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    
    return feature_columns

def prediction_input_fn(features, batch_size=64):
    """An input function for prediction."""
    # Convert the inputs to a dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

def number_to_label(num, y_train):
    y_vocab = list(y_train.unique())
    return y_vocab[num]

def main():

    train, test, validate, y_train, y_test, validate_y = data_prep(dfAll, label)
    
    # print(train.head)
    # print(y_train.head)
    # print(type(y_train))
    # y_vocab = list(y_train.unique())
    # print(y_vocab)
    
    #create feature columns, labels have to be numeric
    y_train = convert_labels_to_num(y_train)
    y_test = convert_labels_to_num(y_test)
    validate_y = convert_labels_to_num(validate_y)
    cat_col, num_col = cols_names(train, cat_or_num = True)
    my_feature_columns = feature_col_creator(cat_col, num_col, train)
    # print(my_feature_columns)
    
    # Build a DNN with 1 hidden layer
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Can have multiple hidden layers, node count between input and output layer size
        hidden_units=[4],
        # The model must choose between n_classes.
        n_classes=4)
    
    # Train the model
    classifier.train(
        input_fn=lambda: input_fn(train, y_train, training=True),
        steps=15000)
    
    # Test the model    
    eval_result = classifier.evaluate(
        input_fn=lambda: input_fn(validate, validate_y, training=False))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    print(eval_result)
    
    # Validate the model
    prediction = classifier.predict(
        input_fn=lambda: prediction_input_fn(test))
    
    y_vocab = list(y_train.unique())
    
    pred_acc_count = 0
    for pred_dict, expec in zip(prediction, y_test):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
            y_vocab[class_id], 100 * probability, number_to_label(expec, y_test)))
        if y_vocab[class_id] == number_to_label(expec, y_test):
            pred_acc_count += 1
    print('Prediction accuracy of validation is {}%' 
              .format(100*(pred_acc_count/len(y_test))))

#%%
if __name__ == "__main__":
    main()