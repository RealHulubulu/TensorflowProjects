# TensorflowProjects

These are projects I have been developing using Tensorflow Estimators. I am using the tutorials found at https://www.tensorflow.org/tutorials
as a base for my projects. I want to create simple apps that utilize Tensorflow Estimators. This will allow for people to explore the power behind machine learning, and specifically neural networks, in data analytics. All links are also found at the bottom of this file.
Use Google Colab for a free environment where you can explore and test my code and tensorflow in general. It's pretty awesome.

Update 6/14/2020: <br /> I added in two scripts that use DNNClassifier. The TensorflowDNNClassifier.py creates a model using a cars
dataset from UC Irvine. The dataset isn't ideal for deep learning as it is simpler and linear. If you have a more complex dataset it
is very simple to use this code on your dataset. The TensorflowDNNClassifier_MNIST.py creates a model for recognizing numbers using
the famous MNIST dataset from keras. I got the general code from a tutorial. It is very simple code at around 50 lines but creates, 
trains, and tests a deep learning model on image detection. Yeah APIs! Links at the bottom for tutorials and such.

Update 6/10/2020: <br /> I added in three scripts. They all use the Estimator linear classifier found in the tutorials for Tensorflow. The tensorflowLinearEstimator.py
is just the tutorial but with some added crossed columns. The TensorflowEstimatorLinearClassifier_Cars_Dataset.py uses the same code from the
tutorial but on a different dataset from UC Irvine. The TensorflowEstimatorLinearClassifier_App.py is a general solution that works with any 
csv file with classification data. Currently it has 7 datasets you can try out and a custom option for adding in your own custom dataset. The
datasets are either from the tutorials for Tensorflow or from UC Irvine at https://archive.ics.uci.edu/ml/datasets.php.

Useful links

Google Colab
https://colab.research.google.com/

Tensorflow Tutorials:
https://www.tensorflow.org/tutorials

Estimator Linear Classification Tutorial:
https://www.tensorflow.org/tutorials/estimator/linear

Estimator DNN Classifier Tutorial:
https://www.tensorflow.org/tutorials/estimator/premade

Estimator DNN Classifier MNIST Tutorial:
https://codeburst.io/use-tensorflow-dnnclassifier-estimator-to-classify-mnist-dataset-a7222bf9f940

Datasets:
https://archive.ics.uci.edu/ml/datasets.php
