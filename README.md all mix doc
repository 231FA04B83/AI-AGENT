
# Healthcare and AI Projects

This repository contains three machine learning and deep learning projects:

1. **Healthcare Disease Prediction Using Machine Learning**
2. **IMDB Sentiment Analysis Using Deep Learning**
3. **Dog and Cat Image Classification Using CNN**

---

## Project 1: Healthcare Disease Prediction Using Machine Learning

### 📌 Project Content
Develop a machine learning model to predict possible diseases based on patient health-related data.

### 🧠 Key Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Pickle

### 📝 Description
This project uses logistic regression to predict diseases based on features such as symptoms, age, and gender. The dataset is preprocessed, label encoded, and used to train the model.

### 💻 Code
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

df = pd.read_csv("healthcare_dataset.csv")
df.drop(columns=['Name'], inplace=True)
label_encoders = {col: LabelEncoder().fit(df[col]) for col in df.select_dtypes(include=['object']).columns}
for col, le in label_encoders.items():
    df[col] = le.transform(df[col])

X = df.drop("Disease", axis=1)
y = df["Disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

with open('disease_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### 📈 Output
- Accuracy score of the model.
- Classification report.
- A `.pkl` file to be deployed in healthcare systems.

### 🔬 Further Research
- Expand dataset
- Try ensemble models
- Build a web interface
- Integrate real-time data

---

## Project 2: IMDB Sentiment Analysis Using Deep Learning

### 📌 Project Content
Classify IMDB movie reviews as positive or negative using an LSTM-based deep learning model.

### 🧠 Key Technologies
- Python
- NLTK, BeautifulSoup
- Keras/TensorFlow
- Pandas

### 📝 Description
Movie reviews are cleaned, tokenized, and converted into padded sequences. An LSTM network learns to associate text with sentiment.

### 💻 Code
```python
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

nltk.download('stopwords')
movie_reviews = pd.read_csv("/content/IMDB Dataset.csv")

def preprocess(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text().lower()
    tokens = [word for word in text.split() if word not in stopwords.words('english')]
    return ' '.join(tokens)

movie_reviews['cleaned'] = movie_reviews['review'].apply(preprocess)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(movie_reviews['cleaned'])
sequences = tokenizer.texts_to_sequences(movie_reviews['cleaned'])
X = pad_sequences(sequences, maxlen=200)
y = pd.get_dummies(movie_reviews['sentiment']).values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Embedding(5000, 64, input_length=200),
    LSTM(64),
    Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```

### 📈 Output
- Accuracy of the model.
- Predicts new reviews.
- Can be expanded to batch predictions.

### 🔬 Further Research
- Use pre-trained embeddings
- Try GRU or Bi-LSTM
- Deploy using APIs
- Add attention mechanisms

---

## Project 3: Dog and Cat Image Classification Using CNN

### 📌 Project Content
A CNN model to classify whether an image contains a dog or a cat.

### 🧠 Key Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

### 📝 Description
The dataset is split into training and validation sets with normalization and resizing. A CNN is trained using binary crossentropy.

### 💻 Code
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

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

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
```

### 📈 Output
- Accuracy and loss plots
- Image classification results
- Final model accuracy

### 🔬 Further Research
- Use ResNet/Inception
- Tune hyperparameters
- Use transfer learning
- Expand to more animal types
- Deploy using Flask or Streamlit
