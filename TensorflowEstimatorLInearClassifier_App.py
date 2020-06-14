#!/usr/bin/env python
# coding: utf-8
"""
This code is from Tensorflow tutorials. It has some modifications. This script is a general app
that takes an input of a csv formatted dataset and outputs the accuracy and predictions of a 
trained model. The dataset is automatically split among training/testing/validation. You can 
try out 7 different datasets found online or add in your own custom one. The custom entry needs
improving and only works with online datasets.

Link to the tutorial: https://www.tensorflow.org/tutorials/estimator/linear
Link to the general tutorials that use the googleapis datasets: https://www.tensorflow.org/tutorials
Link to UC Irvine database resource: https://archive.ics.uci.edu/ml/datasets.php
"""
import sys
import pandas as pd
from IPython.display import clear_output
import tensorflow as tf
from sklearn.model_selection import train_test_split

base_accuracy_list = []
validation_list = []

#%%
def user_input():
    """user input function for selecting dataset, label from dataset, number of runs (defaults to 1 run)"""
    print("Dataset Options\n 1-flowers\n 2-titanic\n 3-breast_cancer\n 4-adult_income\n 5-cars\n 6-chess\n 7-mushrooms\n 8-custom\n else-EXIT")
    print()
    data_selection = input("Enter dataset number:")
    print("Dataset selected is: " + data_selection)
    print()
    if data_selection == str(1):
        # Load dataset if file split between train and eval
        CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
        dftrain = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv", names=CSV_COLUMN_NAMES, header=0)
        dfeval = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv", names=CSV_COLUMN_NAMES, header=0)
        label = "Species"
        frames = [dftrain, dfeval]
        df = pd.concat(frames)
    elif data_selection == str(2):
        dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
        dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
        label = "survived"
        frames = [dftrain, dfeval]
        df = pd.concat(frames)
    elif data_selection == str(3):
        CSV_COLUMN_NAMES = ["Class", "age", "menopause", "tumor-size", "inv-nodes", "nodes-caps", "deg-malig", "breast", "breast-quad", "irradiat"] #breast-cancer.names
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data', names=CSV_COLUMN_NAMES, header=0)
        label = "Class"
    elif data_selection == str(4):
        CSV_COLUMN_NAMES = ["income", "age", "workclass", "fnlwgt", "education", "education-num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital-loss", "hours-per-week", "native-country"] #adult.names
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', names=CSV_COLUMN_NAMES, header=0)
        label = "income"
    elif data_selection == str(5):       
        CSV_COLUMN_NAMES = ["Buying", "Maint", "Doors", "Persons", "Lug_Boot", "Safety", "Classif"] #car.names
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=CSV_COLUMN_NAMES, header=0)
        label = "Classif"
    elif data_selection == str(6):
        CSV_COLUMN_NAMES = ["white-king-file", "white-king-rank", "white-rook-file", "white-rook-rank", "black-king-file", "black-king-rank", "optimal-depth-of-win"] #krkopt.info
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data', names=CSV_COLUMN_NAMES, header=0)
        label = "optimal-depth-of-win"
    elif data_selection == str(7):
        CSV_COLUMN_NAMES = ["edible_poison", "cap-shape", "cap-surface", "cap-color", "bruisesTF", "odor", "gill-attachment",
                            "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                            "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                            "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"] #agaricus-lepiota.names
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data', names=CSV_COLUMN_NAMES, header=0)
        label = "edible_poison"
    elif data_selection == str(8):
        try:
            df_filepath = input("Enter custom data.csv file path:")
            are_col_named = input("Are columns already named?:")
            if are_col_named == "n" or are_col_named == "N" or are_col_named == "no" or are_col_named == "No":
                column_names = input("Enter column names w/ spaces between (ex- class height weight color ...):")
                CSV_COLUMN_NAMES = Convert(column_names)
                df = pd.read_csv(df_filepath, names = CSV_COLUMN_NAMES, header = 0)
            else:
                df = pd.read_csv(df_filepath)
        except:
            print("ERROR, just gonna exit... start it over again")
            sys.exit()
    else:
        print("Exiting...")
        sys.exit()
        
                     
    dfAll = df
    label = label
    try:
      number_of_runs = input("How many runs?:")
      number_of_runs = int(number_of_runs)
      print("Number of runs is: " + str(number_of_runs))
    except:
      print("An exception occurred, not int input, setting runs to 1")
      number_of_runs = 1
    # number_of_runs = int(input("How many runs?:"))
    # print("Number of runs is: " + number_of_runs)
    print()
    # base_accuracy_list = []
    # validation_list = []
    print("Below is a look at the data in the selected dataset using df.head()")
    print(dfAll.head())
    print()
    print("This will run {} times and display the average accuracy and valdiation at the end".format(number_of_runs))
    input("Press Enter to continue...")
    
    return (dfAll, label, number_of_runs)
#%%
def Convert(string): 
    """converts string to list"""
    li = list(string.split(" ")) 
    return li 

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

def split_data(data, test_size=0.2):
    """splits data into training (80%) and testing (20%)"""
    train, test = train_test_split(data, test_size=test_size)
    return train, test

def cols_names(data, cat_or_num = False):
    """returns lists of column names, can handle datasets with both categorical and numeric data"""
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

def feature_col_creator(cat_col, num_col, train_data):
    """returns the feature columns needed for building the model"""
    feature_columns = []
    
    for feature_name in cat_col:
      vocabulary = train_data[feature_name].unique()
      # print(vocabulary)
      
      feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    
    for feature_name in num_col:
      feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    
    return feature_columns
    
def pred_input_fn(features, batch_size=32):
    """creates a separate input function just for validation"""
    """An input function for prediction"""
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
    
def linear_est_creator(feature_columns, num_classes, y_vocab):
    """returns a created LinearClassifier model"""
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                               n_classes=num_classes, label_vocabulary=y_vocab)
    return linear_est

def data_prep(data, y_label_name):
    """takes in the dataset and label, returns splits for training, testing, validation"""
    train, test_no_valid = split_data(data)
    test, validate = split_data(test_no_valid, test_size=0.05)
    
    # #doing pop below removes Classif col
    y_train = train.pop(y_label_name)
    y_test = test.pop(y_label_name) 
    validate_y = validate.pop(y_label_name)
    
    #y labels have to be strings to use y vocab
    y_train = y_train.astype(str)
    y_test = y_test.astype(str)
    validate_y = validate_y.astype(str)
    
    return train, test, validate, y_train, y_test, validate_y
   
def linear_class_est_train_test_valid(y_label_name, dfAll):
    """takes in dataset and label, uses prevoius functions for training, test, validation"""
    train, test, validate, y_train, y_test, validate_y = data_prep(dfAll, y_label_name)
    
    CATEGORICAL_COLUMNS, NUMERIC_COLUMNS = cols_names(train, cat_or_num = True)
    feature_columns = feature_col_creator(CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, train)
    
    y_vocab = list(y_test.unique())
    num_classes = len(y_vocab)
    
    train_input_fn = make_input_fn(train, y_train)
    test_input_fn = make_input_fn(test, y_test, num_epochs=1, shuffle=False)
    
    linear_est = linear_est_creator(feature_columns, num_classes, y_vocab)
    
    training = linear_est.train(train_input_fn)
    result = linear_est.evaluate(test_input_fn)
    
    clear_output()
    
    #Below is for accuracy
    base_accuracy_list.append(result['accuracy'])
    # print(result)
    # print(result['accuracy'])
    
    #Below is for validations 
    prediction_results(linear_est, validate, validate_y, y_vocab)

def prediction_results(linear_est, validate, validate_y, y_vocab):
    """saves validation results for each run in a list"""
    prediction = linear_est.predict(input_fn=lambda: pred_input_fn(validate))
    temp_list_per_run = []
    for pred_dict, expec in zip(prediction, validate_y):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        temp_list_per_item = [y_vocab[class_id], str(100*probability), expec]
        temp_list_per_run.append(temp_list_per_item)
    validation_list.append(temp_list_per_run)
    
# Use this function if validating against new data
# def prediction_new_results(linear_est, y_vocab):
#     validate = "some user input data as pd"
#     validate_y = "some user input data labels as pd"
#     prediction = linear_est.predict(input_fn=lambda: pred_input_fn(validate))
#     temp_list_per_run = []
#     for pred_dict, expec in zip(prediction, validate_y):
#         class_id = pred_dict['class_ids'][0]
#         probability = pred_dict['probabilities'][class_id]
#         temp_list_per_item = [y_vocab[class_id], str(100*probability), expec]
#         temp_list_per_run.append(temp_list_per_item)
#     validation_list.append(temp_list_per_run)

def main():
    """puts everything together"""
    dfAll, label, number_of_runs = user_input()
    
    for i in range(number_of_runs):
        linear_class_est_train_test_valid(label, dfAll)  
    print()
    
    #To see avg acc across all runs    
    print("Accuracy")
    # print(base_accuracy_list)
    base_avg_acc = sum(base_accuracy_list)/len(base_accuracy_list)    
    print("Avg Acc: " + str(base_avg_acc))
    
#To see validation for the final run   
    print("Validation")
    # print(validation_list)
    pred_accuracy = 0
    for index in validation_list[-1]:
        y_label = index[0]
        prob = index[1]
        expect = index[2]
        print('Prediction is "{}" ({}%), expected "{}"'.format(
                y_label, prob, expect))
        if y_label == expect:
            pred_accuracy += 1
    print('Prediction accuracy of validation from final run {} is {}%' 
          .format(validation_list.index(validation_list[-1]),
                  100*(pred_accuracy/len(validation_list[-1]))))
    
# #To see validation testing for each run
# print("Validation")
# for run in validation_list:
#     pred_accuracy = 0
#     for index in run:
#         y_label = index[0]
#         prob = index[1]
#         expect = index[2]
#         print('Prediction is "{}" ({}%), expected "{}"'.format(
#             y_label, prob, expect))
#         if y_label == expect:
#             pred_accuracy += 1
#     print('Prediction accuracy of validation from run {} is {}%' .format(validation_list.index(run),100*(pred_accuracy/len(run))))
#     print()
#     print()
#%%
if __name__ == "__main__":
    main()
