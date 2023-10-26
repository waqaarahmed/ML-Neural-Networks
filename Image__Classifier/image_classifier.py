import os
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

#dividing training_images and testing_images by 255
x_train = x_train / 255
x_test = x_test / 255

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

"""
#Section: Use this code once to train the system
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(classes[y_train[i][0]])

plt.show()

#optional: decreasing the number of images in neural network
# x_train = x_train[:10000]
#y_train = y_train[:10000]
#x_test = x_test[:2000]
#y_test = y_test[:2000]



#Building a neural network
#1: Defining the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # --> Defining an input layer
model.add(layers.MaxPooling2D((2, 2))) #--> Adding maxpooling laying after conv2d layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten()) #--> Flattening(making a matrix flat for e.g making a 3x3 into 9) the input we get from the above modelI.
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) #--> The final layer(output layer)

#2: Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#3: Fit the model
model.fit(x_train, y_train, epochs=10, validation_data=[x_test, y_test])

#4: Testing and evaluating the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

#5: Saving the model
model.save('image_classifier.model')

#End of Section"""
model = models.load_model('image_classifier.model')

#reading image from the folder
image = mpimg.imread(os.path.join('folder_name', 'file_name'))
#image = cv2.imread(image)

#converting image color fo cv2
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image, cmap=plt.cm.binary)
#passing the image as numpy array and dividing it by 255
predict = model.predict(np.array([image]) / 255)
index = np.argmax(predict)
print(f"Image is of a: {classes[index]}")
