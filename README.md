# Implementation-of-CNN

## AIM

To Develop a convolutional deep neural network for digit classification.

## Problem Statement and Dataset

The goal of this project is to develop a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset. The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9), totaling 60,000 training images and 10,000 test images.

![image](https://github.com/user-attachments/assets/63ef8f17-6f7f-4896-a957-bf33012ff527)

## Neural Network Model

![image](https://github.com/user-attachments/assets/beb8f852-ffa2-4b72-baf0-c25bae596d3c)


## DESIGN STEPS

### STEP 1:
Import the necessary libraries and load the dataset

### STEP 2:
Reshape and normalize the data

### STEP 3:
Create the EarlyStoppingCallback function

### STEP 4:
Create the convulational model and compile the model

### STEP 5:
Train the model

## PROGRAM

### Name:Kavinesh M
### Register Number:212222230064
```py
import numpy as np
import tensorflow as tf

data_path ="mnist.npz"

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=data_path)

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")

def reshape_and_normalize(images):
  
    # Reshape the images to add an extra dimension (at the right-most side of the array)
    images = images.reshape(60000,28,28,1)
    # Normalize pixel values
    images = images/255
    return images
# Reload the images in case you run this cell multiple times
(training_images, _), _ = tf.keras.datasets.mnist.load_data(path=data_path)
# Apply your function
training_images = reshape_and_normalize(training_images)
print('Name: Kavinesh M RegisterNumber: 212222230064 \n')
print(f"Maximum pixel value after normalization: {np.max(training_images)}\n")
print(f"Shape of training set after reshaping: {training_images.shape}\n")
print(f"Shape of one image after reshaping: {training_images[0].shape}")

# Remember to inherit from the correct class
class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    # Define the correct function signature for on_epoch_end method
    def on_epoch_end(self,epochs,logs=None):

        # Check if the accuracy is greater or equal to 0.995
        if logs['accuracy'] >= .995:

            # Stop training once the above condition is met
            self.model.stop_training = True

            print("\n\nReached 99.5% accuracy so cancelling training!\n")
            print('Name: Kavinesh M  Register Number: 212222230064\n')

#convolutional_model

def convolutional_model():
    """Returns the compiled (but untrained) convolutional model.

    Returns:
        tf.keras.Model: The model which should implement convolutions.
    """

    # Define the model
    model = tf.keras.models.Sequential([ 
            tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
    ]) 
    # Compile the model
    model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
          
    return model
# Define your compiled (but untrained) model
model = convolutional_model()
training_history = model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStoppingCallback()])
```

## OUTPUT

### Reshape and Normalize output

![image](https://github.com/user-attachments/assets/ad1583b0-a273-491c-9be9-3f03c6766df2)


### Training the model output

![image](https://github.com/user-attachments/assets/c6e598e6-3308-41ac-bd9b-620a733fe247)




## RESULT
Hence a convolutional deep neural network for digit classification was successfully developed.
