# Dog and Cat Image Classification Using CNN

---

## 1. Project Overview

Image classification is a core task in computer vision, involving categorizing images into predefined classes. This project implements a Convolutional Neural Network (CNN) to classify images as either dogs or cats.

The model learns to extract features from images and distinguish between the two categories, showcasing a practical application of deep learning in image recognition.

---

## 2. Dataset and Preprocessing

### Dataset Description

- The dataset contains labeled images of dogs and cats, commonly sourced from the Kaggle "Dogs vs Cats" dataset.
- Images vary in size, lighting, and background, which requires standardization.
- Typical dataset size: ~25,000 images (12,500 dog and 12,500 cat images).

### Preprocessing Steps

- **Resizing:** Images resized to 150x150 pixels to maintain uniform input size.
- **Normalization:** Pixel values scaled from [0, 255] to [0, 1] by dividing by 255.
- **Data Splitting:** An 80:20 split is used for training and validation data.
- **Augmentation (optional):** Techniques like rotation, zoom, and flipping can be applied to increase dataset variety and reduce overfitting.

---

## 3. CNN Architecture: Theory and Design

### Why CNNs?

CNNs effectively capture spatial features in images through:

- **Convolutional layers** that learn filters to detect edges, shapes, textures.
- **Pooling layers** that reduce dimensionality and emphasize dominant features.
- **Fully connected layers** that classify extracted features into labels.

### Model Structure

| Layer           | Parameters | Output Shape      |
|-----------------|------------|-------------------|
| Conv2D (32)     | 896        | (148, 148, 32)    |
| MaxPooling2D    | 0          | (74, 74, 32)      |
| Conv2D (64)     | 18,496     | (72, 72, 64)      |
| MaxPooling2D    | 0          | (36, 36, 64)      |
| Conv2D (128)    | 73,856     | (34, 34, 128)     |
| MaxPooling2D    | 0          | (17, 17, 128)     |
| Flatten         | 0          | (36992)           |
| Dense (512)     | 18,903,040 | (512)             |
| Dense (1)       | 513        | (1)               |

---

## 4. Data Preparation Code

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
    'path_to_dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)
