# -*- coding: utf-8 -*-

"""
This code uses some of a tutorial I found online for using DNNClassifier on the MNIST dataset. Here 
I use the MNIST Fashion dataset using keras. 

With current parameters
    num_models_to_test = 5, 
    hyper_param_list = [0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.00008]
the runtime is about 3089 seconds

Link to the tutorial: https://codeburst.io/use-tensorflow-dnnclassifier-estimator-to-classify-mnist-dataset-a7222bf9f940 
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from datetime import datetime


fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
# print(fashion_mnist)
num_models_to_test = 5
hyper_param_list = [0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.00008]

def prep_data(data_set, valid_size = 0.10):
    
    # uses labeled data split into training and testing sets
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    # creates train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(train_images, train_labels,
                                                            test_size=valid_size, 
                                                            # stratify to remove bias in distribution
                                                            # fashion_mnist is cleaned already
                                                            stratify=train_labels)
    
    return X_train, X_valid, y_train, y_valid, test_images, test_labels

def feature_col_creator(data):
    dim1 = len(data[0][0][0][0]) #28
    dim2 = len(data[0][0][0]) # 28
    feature_columns = [tf.feature_column.numeric_column("x", shape=[dim1, dim2])]
    return feature_columns

def create_classifier(feature_columns, hidden_1, 
                      hidden_2, optimizer, n_classes, dropout ):
    classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            #these two hidden layer sizes are between the input and output layer sizes of 784 and 10
            hidden_units=[hidden_1, hidden_2], 
            optimizer = optimizer,
            n_classes=n_classes,
            dropout=dropout
        )
    
    return classifier

def convert_input(dataset):
    return dataset.astype(np.int32)

def make_input_function(x_values, y_values, batch_size, num_epochs, isTraining):
    def input_function():
        model_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
                x={"x": convert_input(x_values)},
                y=convert_input(y_values),
                num_epochs=num_epochs,
                batch_size=batch_size,
                shuffle=isTraining
            )
        return model_input_fn
    return input_function()

def prediction_output(prediction, test_labels):
    pred_accuracy = 0 
    pred_output_list = []
    print()
    print("Testing...")
    for pred_dict, expec in zip(prediction, test_labels):
        run_list = []
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        # print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        # class_id, 100 * probability, expec))
        if class_id == expec:
            pred_accuracy += 1
        run_list.extend((class_id, probability, expec))
        pred_output_list.append(run_list)
    # print('Prediction accuracy of test set is {}%' .format(100*(pred_accuracy/len(test_labels))))
    return 100*(pred_accuracy/len(test_labels)), pred_output_list

# reads in dataset and splits into training and test, training has validation within it
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
# run_valid_list = []
# run_test_list = []

# print(type(train_labels))

print(len(fashion_mnist[0][0][0][0]))
print(len(fashion_mnist[0][0][0]))

#%%
# to see what each pic looks like with plot
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(class_names)

# plt.figure()
# plt.imshow(train_images[3])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# print(class_names[train_labels[3]])
#%%

# num_models_to_test = 2
# hyper_param_list = [0.001, 0.0001]

num_models_to_test = 5
hyper_param_list = [0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.00008]
run_valid_list = []
run_test_list = []
pred_list = []

start_time = time.time()
training_time = 0.0

for i in range(len(hyper_param_list)):
    this_run_valid_list = []
    this_run_test_list = []
    this_run_pred_list = []
    for j in range(num_models_to_test):
        
        # https://keras.io/api/optimizers/
        optimizer_adam = tf.compat.v1.train.AdamOptimizer(learning_rate=hyper_param_list[i])

        X_train, X_valid, y_train, y_valid, test_images, test_labels = prep_data(fashion_mnist)
        
        feature_columns = feature_col_creator(fashion_mnist)
        
        classifier = create_classifier(feature_columns=feature_columns, hidden_1=256, hidden_2=32,
                                       optimizer=optimizer_adam, n_classes=10, dropout = 0.1)
        
        train_input_function = make_input_function(X_train, y_train, batch_size = 128, 
                                                   num_epochs=None, isTraining=True)
        valid_input_function = make_input_function(X_valid, y_valid, batch_size=128, 
                                                   num_epochs=1, isTraining=False)
        test_input_function = make_input_function(test_images, test_labels, batch_size=128, 
                                                  num_epochs=1, isTraining=False)
        
        training = classifier.train(input_fn=train_input_function, steps=50000) #50000
        
        end_time = time.time()
        training_time = end_time - start_time
        
        validation = classifier.evaluate(input_fn=valid_input_function)["accuracy"]
        this_run_valid_list.append(validation)
        # print("\nValidation Accuracy: {0:f}%\n".format(validation*100))
        testing_score = classifier.evaluate(input_fn=test_input_function)["accuracy"]
        this_run_test_list.append(testing_score)
        # print("\nTest Accuracy: {0:f}%\n".format(testing_score*100))
        print()
        prediction = classifier.predict(input_fn = test_input_function)
        prediction_acc, this_run_pred_list = prediction_output(prediction, test_labels)
        
        print("Validation Accuracy: {0:f}%\n".format(validation*100))
        print("Test Accuracy: {0:f}%\n".format(testing_score*100))
        print("Predict Accuracy: {0:f}%\n".format(prediction_acc))
        
        
    run_valid_list.append(this_run_valid_list)
    run_test_list.append(this_run_test_list)
    pred_list.append(this_run_pred_list)
    
print("Validation Acc per run: " + str(run_valid_list)) 
print("Test Acc per run: " + str(run_test_list))


end_time = time.time()
run_time = end_time - start_time

print()
print("Training/Validation/Testing Time: {}".format(run_time))
print()

model_scores = []

for i in range(len(hyper_param_list)):
    print("Model {}".format(i+1))
    print("All Validation: {}".format(run_valid_list[i]))
    print("Avg Validation: {}".format(sum(run_valid_list[i])/len(run_valid_list[i])))
    print("All Test: {}".format(run_test_list[i]))
    print("Avg Test: {}".format(sum(run_test_list[i])/len(run_test_list[i])))
    model_scores.append(sum(run_test_list[i])/len(run_test_list[i]))
    print()
print("Best score of {} from {}".format(max(model_scores), model_scores.index(max(model_scores))+1))

index_of_best = model_scores.index(max(model_scores))
log_file_info_headers = ["hyper_param_list", "index_of_best", "score_of_best", "training_time", "run_time"]
log_file_info = [hyper_param_list, index_of_best, max(model_scores), training_time, run_time]
now = str(datetime.now()).replace(" ", "_").replace(":", "")

with open("mnist_fashion_log_file."+now+".txt", "a+") as f:
            f.write(str(log_file_info_headers) + "\n")
            f.write(str(log_file_info) + "\n")
            for list_item in pred_list[index_of_best]:
                f.write('%s\n' % list_item)
            print("done!!!")
#%%
# # TESTING AREA
            
# # print(prediction)
# prediction = list(classifier.predict(input_fn = test_input_function))
# print()
# # validation = classifier.evaluate(input_fn=valid_input_function)
# # print()
# # print()
# # print(validation)

# print(prediction[0]['probabilities'][0])
# print()
# print(prediction[0])

# training_predictions = ([item['probabilities'] for item in prediction])
# print()
# print(type(training_predictions))
# print()
# # print(training_predictions)
# for preds in training_predictions:
#     max_pred = max(preds)
#     max_pred_index = np.argmax(preds)
#     # print(max_pred, max_pred_index)
    
# print(test_labels[0])
# print(np.argmax(prediction[0]['probabilities']))
   

# pred_accuracy = 0 
# print()
# print("Testing...")
# for pred_dict, expec in zip(prediction, test_labels):
#     class_id = pred_dict['class_ids'][0]
#     probability = pred_dict['probabilities'][class_id]
#     print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
#         class_id, 100 * probability, expec))
#     if class_id == expec:
#         pred_accuracy += 1
# print('Prediction accuracy of test set is {}%' .format(100*(pred_accuracy/len(test_labels))))

#%%
# # 0.0006 comes up as best learning_rate
# # hyper_param_list = [0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.00008]

# # https://keras.io/api/optimizers/
# # optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

# start_time = time.time()

# for i in range(len(hyper_param_list)):
    
#     this_run_valid_list = []
#     this_run_test_list = []
#     for j in range(num_models_to_test):
        
#         optimizer_adam = tf.keras.optimizers.Adam(learning_rate=hyper_param_list[i])
        
#         # creates train and validation sets
#         X_train, X_valid, y_train, y_valid = train_test_split(train_images, train_labels,
#                                                             test_size=0.10, 
#                                                             # stratify to remove bias in distribution
#                                                             stratify=train_labels)
        
#         # define one feature with shape 28x28
#         feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]
#         print(feature_columns)
        
#         # this just converts dataset into correct input type for DNN
#         def input(dataset):
#             return dataset.astype(np.int32)
        
#         # Build 2 layer DNN classifier
#         classifier = tf.estimator.DNNClassifier(
#             feature_columns=feature_columns,
#             #these two hidden layer sizes are between the input and output layer sizes of 784 and 10
#             hidden_units=[256, 32],
            
            
#             # optimizer = 'Adam',
            
#             optimizer = optimizer_adam,
            
#             n_classes=10,
#             dropout=0.1
#         )
        
#         # Create training input function
#         train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
#             x={"x": input(X_train)},
#             y=input(y_train),
#             num_epochs=None,
#             batch_size=50,
#             shuffle=True
#         )
#         # Train the model, note there are 6000 samples in mnist dataset
#         classifier.train(input_fn=train_input_fn, steps=50000) #50000
        
#         # Create validation input function
#         validate_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
#             x={"x": input(X_valid)},
#             y=input(y_valid),
#             num_epochs=1,
#             shuffle=False
#         )
#         # Validate the model
#         accuracy_score = classifier.evaluate(input_fn=validate_input_fn)["accuracy"]
#         print("\nValidation Accuracy: {0:f}%\n".format(accuracy_score*100))
#         this_run_valid_list.append(accuracy_score)
    
#         # Create testing input function
#         test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(        
#             x={"x": input(test_images)},
#             y=input(test_labels),
#             num_epochs=1,
#             shuffle=False
#         )
#         # Test the model
#         testing_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
#         print("\nTest Accuracy: {0:f}%\n".format(testing_score*100))
#         this_run_test_list.append(testing_score)
    
#     run_valid_list.append(this_run_valid_list)
#     run_test_list.append(this_run_test_list)
    
# print("Validation Acc per run: " + str(run_valid_list)) 
# print("\nTest Acc per run: " + str(run_test_list))

# end_time = time.time()
# run_time = end_time - start_time

# print()
# print("Training Time: {}".format(run_time))
# print()

# model_scores = []

# for i in range(len(hyper_param_list)):
#     print("Model {}".format(i+1))
#     print("All Validation: {}".format(run_valid_list[i]))
#     print("Avg Validation: {}".format(sum(run_valid_list[i])/len(run_valid_list[i])))
#     print("All Test: {}".format(run_test_list[i]))
#     print("Avg Test: {}".format(sum(run_test_list[i])/len(run_test_list[i])))
#     model_scores.append(sum(run_test_list[i])/len(run_test_list[i]))
#     print()

# print("Best score of {} from {}".format(max(model_scores), model_scores.index(max(model_scores))+1))
