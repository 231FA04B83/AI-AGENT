# Dog and Cat Image Classification Using CNN

---

## 1. Project Content

### Overview

This project builds a Convolutional Neural Network (CNN) to classify images into two categories: **Dogs** and **Cats**. It demonstrates deep learning techniques applied to image recognition, using a popular dataset of dog and cat images.

### Dataset Details

- Approximately 25,000 images (12,500 dogs and 12,500 cats).
- Images vary in size, lighting, and background.
- Stored in folders: `/dogs` and `/cats`.

### Folder Structure


### Project Stages

1. Data loading and preprocessing  
2. CNN model creation  
3. Model training  
4. Model evaluation and visualization  
5. Model saving and inference

## 2. Project Code

### Data Preparation

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)
CNN Model Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

Model Training

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)
Visualizing Training Results
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14,6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()



3. Key Technologies

Python: Main programming language.

TensorFlow & Keras: Deep learning framework for CNN construction and training.

ImageDataGenerator: For loading, rescaling, and augmenting images.

Matplotlib: For plotting training and validation metrics.

Jupyter Notebook: Interactive environment used to develop the project.

4. Description

Problem Statement
Build a model that automatically classifies images as either a dog or a cat.

Why CNN?
Convolutional Neural Networks are specialized for images. They learn spatial hierarchies of features automatically, eliminating the need for manual feature extraction.

Model Architecture Summary
Convolutional layers to detect edges, textures, and shapes.

MaxPooling layers to reduce dimensionality and overfitting.
Fully connected layers to interpret the features and classify images.

Training Details
Dataset split: 80% training, 20% validation.

Loss function: Binary Crossentropy (for binary classification).

Optimizer: Adam (adaptive learning rate).

Epochs: 10 (can be increased for better results).

Batch size: 32.

5. Output

Metrics Achieved
Metric	Description
Training Accuracy	Accuracy on the training dataset images.
Validation Accuracy	Accuracy on unseen validation images.
Training Loss	Loss value during training, indicating error.
Validation Loss	Loss on validation data, indicates overfitting.

Visual Output
Graphs showing training & validation accuracy and loss over epochs.

Model predictions classifying new images as dog or cat.

Sample Prediction Code
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = 'path_to_test_image.jpg'
img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0] > 0.5:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")

6. Further Research
Improving the Model
Data Augmentation: Adding flips, rotations, zooms to increase data diversity.

Transfer Learning: Using pre-trained models like VGG16, ResNet for better accuracy and training efficiency.

Hyperparameter Tuning: Experiment with batch size, learning rates, optimizers.

Expanding the Project
Multi-class classification of dog breeds and cat breeds.

Object detection to localize animals in images.

Deploy as a web app using Flask or Streamlit.

Advanced Techniques
Attention mechanisms for better feature focus.

Explainability methods like Grad-CAM to visualize what model learns.

Generative models to create synthetic images for augmentation.
