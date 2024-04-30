# Neural Network for Handwritten Digit Recognition

This Python script demonstrates how to build and train a neural network for handwritten digit recognition using the MNIST dataset. The model is implemented using TensorFlow and Keras.

## Requirements

- Python 3.10
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

## Example

```
import tensorflow as tf
from tensorflow import keras

# Load the MNIST dataset
ndata = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = ndata.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255 , x_test / 255

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate model accuracy
model.evaluate(x_test, y_test)
```
## Note

- This script provides a basic implementation of a neural network for handwritten digit recognition.
- Adjustments to the model architecture, hyperparameters, and dataset augmentation may improve performance.
