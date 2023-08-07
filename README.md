# Number_Recognition
## MNIST Digit Classification using Keras
This repository demonstrates a neural network-based approach to classify handwritten digits from the famous MNIST dataset using Keras, a high-level neural networks API running on top of TensorFlow. The project's goal is to train a deep learning model capable of accurately recognizing digits from 0 to 9.   

The digits look like,

![th](https://github.com/Guru02-GiT/Number_Recognition/assets/80115003/c3964b03-4e3e-4345-96be-b14133919e48)

## Getting Started
### Prerequisites
To run the code and experiments, you'll need the following:

  - Python (version 3.6 or later)    
  - Jupyter Notebook (to interact with the provided notebook)  
  - TensorFlow (for deep learning operations)   
  - Keras (for building and training the neural network)   
  - numpy (for array operations)   
  - matplotlib (for visualization)

You can install the required libraries using the following command:   
 ```bash
pip install tensorflow keras numpy matplotlib
```
## Installation
To get started, follow these steps:   
  1. Clone the repository to your local machine:
```bash
git clone https://github.com/Guru02-GiT/Number_Recognition.git
```
  2. Navigate to the project directory:
```bash
cd Number_Recognition
```
  3. Run the Jupyter Notebook file:
```bash
jupyter notebook DeepLearning_MNIST_Classification.ipynb
```
## Procedure 
1. **Dataset Loading**: The MNIST dataset is loaded using Keras, consisting of 60,000 training images and 10,000 test images, each of size 28x28 pixels.  
2. **Data Preprocessing**: Images are reshaped and scaled, and labels are one-hot encoded.  
3. **Model Architecture**: A neural network model with two layers is created using Keras.   
4. **Model Compilation**: The model is compiled using the rmsprop optimizer and categorical crossentropy loss.   
5. **Model Training**: The model is trained for 10 epochs with a batch size of 128.   
6. **Model Evaluation**: After training, the model's performance is evaluated on the test dataset and the predictions and actual labels are also shown.

# Results
  - Upon running the Jupyter Notebook, you'll see the training progress and final evaluation results, including accuracy and the actual and predicted labels for the digits.
    
  - Feel free to experiment with different hyperparameters and model architectures to improve accuracy.
