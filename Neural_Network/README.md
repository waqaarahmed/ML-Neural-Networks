# Neural Network for Handwritten Digit Recognition

This Python script demonstrates how to build and train a neural network for handwritten digit recognition using the MNIST dataset. The model is implemented using TensorFlow and Keras.

## Requirements

- Python 3.x
- TensorFlow
- Keras

You can install the required libraries using pip:

```bash
pip install tensorflow
```

## Usage

1. Ensure you have the necessary dependencies installed.
2. Run the script `neural_network.py`

## Workflow

1. The MNIST dataset is loaded using `tf.keras.datasets.mnist`.
2. The dataset is divided into training and testing sets.
3. The pixel values of the images are normalized to the range [0, 1].
4. The neural network model is defined using Keras Sequential API with the following layers:
  - Flatten: Input layer to flatten the 28x28 pixel images into a 1D array.
  - Dense: Fully connected layer with 128 neurons and ReLU activation function.
  - Dropout: Dropout layer to prevent overfitting.
  - Dense: Output layer with 10 neurons (for 10 classes) and softmax activation function.
5. The model is compiled with the Adam optimizer, sparse categorical crossentropy loss function, and accuracy metrics.
6. The model is trained on the training dataset for a specified number of epochs.
7. The accuracy of the trained model is evaluated on the testing dataset.
