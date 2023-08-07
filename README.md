# Number_Recognition

# MNIST Digit Classification using Keras
## Description
This project demonstrates a neural network-based approach for classifying handwritten digits from the MNIST dataset using Keras, a high-level neural networks API running on top of TensorFlow. The goal is to train a deep learning model to recognize digits from 0 to 9 accurately.
## Procedure
1. Dataset Loading: The MNIST dataset is loaded using the Keras library. The dataset consists of 60,000 training images and 10,000 test images, each of size 28x28 
   pixels.
2. Data Preprocessing: The loaded images are preprocessed by reshaping them to a flat vector of size 28*28 and then scaled to values between 0 and 1 by dividing by 255. Additionally, the labels are one-hot encoded using the `to_categorical` function to prepare them for multiclass classification.
3. Model Architecture: A neural network model is created using the `Sequential` class from Keras. It consists of two layers: a dense hidden layer with 512 neurons and ReLU activation, and an output layer with 10 neurons corresponding to the 10 possible digit classes, using softmax activation.
4. Model Compilation: The model is compiled using the `rmsprop` optimizer and categorical crossentropy loss, as it is a multiclass classification problem. The accuracy metric is also specified to monitor the model's performance during training.
5. Model Training: The model is trained using the `fit` function with the training images and one-hot encoded labels. The training is performed for 10 epochs with a batch size of 128.
6. Model Evaluation: After training, the model's performance is evaluated on the test dataset using the `evaluate` function. The test loss and accuracy are printed to the console.
## Technologies Used
- Python: The programming language used for the implementation of the project.
- Keras: A high-level neural networks API, running on top of TensorFlow, used to build and train the deep learning model.
- TensorFlow: The deep learning framework, on which Keras is built, used for mathematical operations and model execution.
