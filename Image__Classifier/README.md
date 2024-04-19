# Image Classifier using Keras, TensorFlow, Matplotlib, NumPy, and OpenCV

This Python script demonstrates how to build, train, and use an image classifier using deep learning with Keras and TensorFlow. It utilizes the CIFAR-10 dataset for training and testing, and the trained model is saved for later use.

## Usage

Ensure you have the necessary dependencies installed.
Download the CIFAR-10 dataset for training and testing the model.
Run the script image_classifier.py.

### Workflow
The CIFAR-10 dataset is loaded using datasets.cifar10.load_data().
The dataset is preprocessed by normalizing the pixel values to the range [0, 1].
The model architecture is defined using convolutional neural network (CNN) layers with max-pooling and dense layers.
The model is compiled using the Adam optimizer and sparse categorical crossentropy loss.
The model is trained on the training dataset (x_train and y_train) for a specified number of epochs.
Optionally, you can visualize a subset of training images and their corresponding labels.
The trained model is evaluated on the testing dataset (x_test and y_test) to calculate the loss and accuracy.
The trained model is saved for later use.
When using the saved model:
An image is read from the specified folder.
The image is preprocessed and passed through the trained model for prediction.
The predicted class label is displayed based on the highest probability.
Example
An example of using the trained model to classify an image is shown below:

```python
import os
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg

# Load the saved model
model = models.load_model('image_classifier.model')

# Read an image
image = mpimg.imread(os.path.join('folder_name', 'file_name'))

# Convert color for OpenCV
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image, cmap=plt.cm.binary)

# Preprocess the image and make predictions
predict = model.predict(np.array([image]) / 255)
index = np.argmax(predict)
print(f"Image is of a: {classes[index]}")
#### Requirements

- Python 3.x
- Libraries: `tensorflow`, `matplotlib`, `numpy`, `opencv-python`

You can install the required libraries using pip:

```bash
pip install tensorflow matplotlib numpy opencv-python

