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

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
# print(fashion_mnist)
num_models_to_test = 5

# reads in dataset and splits into training and test, training has validation within it
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
run_valid_list = []
run_test_list = []

#%%
# to see what each pic looks like with plot
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure()
plt.imshow(train_images[3])
plt.colorbar()
plt.grid(False)
plt.show()
print(class_names[train_labels[3]])
#%%
# 0.0006 comes up as best learning_rate
hyper_param_list = [0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.00008]

# https://keras.io/api/optimizers/
# optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

start_time = time.time()

for i in range(len(hyper_param_list)):
    
    this_run_valid_list = []
    this_run_test_list = []
    for i in range(num_models_to_test):
        
        optimizer_adam = tf.keras.optimizers.Adam(learning_rate=hyper_param_list[i])
        
        # creates train and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(train_images, train_labels,
                                                            test_size=0.10, 
                                                            # stratify to remove bias in distribution
                                                            stratify=train_labels)
        
        # define one feature with shape 28x28
        feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]
        print(feature_columns)
        
        # this just converts dataset into correct input type for DNN
        def input(dataset):
            return dataset.astype(np.int32)
        
        # Build 2 layer DNN classifier
        classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            #these two hidden layer sizes are between the input and output layer sizes of 784 and 10
            hidden_units=[256, 32],
            
            
            # optimizer = 'Adam',
            
            optimizer = optimizer_adam,
            
            n_classes=10,
            dropout=0.1
        )
        
        # Create training input function
        train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={"x": input(X_train)},
            y=input(y_train),
            num_epochs=None,
            batch_size=50,
            shuffle=True
        )
        # Train the model, note there are 6000 samples in mnist dataset
        classifier.train(input_fn=train_input_fn, steps=50000) #50000
        
        # Create validation input function
        validate_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={"x": input(X_valid)},
            y=input(y_valid),
            num_epochs=1,
            shuffle=False
        )
        # Validate the model
        accuracy_score = classifier.evaluate(input_fn=validate_input_fn)["accuracy"]
        print("\nValidation Accuracy: {0:f}%\n".format(accuracy_score*100))
        this_run_valid_list.append(accuracy_score)
    
    #%%
        # Create testing input function
        test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(        
            x={"x": input(test_images)},
            y=input(test_labels),
            num_epochs=1,
            shuffle=False
        )
        # Test the model
        testing_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
        print("\nTest Accuracy: {0:f}%\n".format(testing_score*100))
        this_run_test_list.append(testing_score)
    
    run_valid_list.append(this_run_valid_list)
    run_test_list.append(this_run_test_list)
    
print("Validation Acc per run: " + str(run_valid_list)) 
print("\nTest Acc per run: " + str(run_test_list))

end_time = time.time()
run_time = end_time - start_time

print()
print("Training Time: {}".format(run_time))
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
