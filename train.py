#importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask,request
import html2text
import pickle

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
layers = keras.layers
models = keras.models


data= pd.read_csv("created_df.csv")

data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

data.label=data.label.astype(str)

data['label'].value_counts()

train_size = int(len(data) * .8)

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

train_cat, test_cat = train_test_split(data['label'], train_size)
train_text, test_text = train_test_split(data['description'], train_size)

max_words =2500
tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)


tokenize.fit_on_texts(train_text)
x_train = tokenize.texts_to_matrix(train_text)
x_test = tokenize.texts_to_matrix(test_text)

encoder = LabelEncoder()
encoder.fit(train_cat)
y_train = encoder.fit_transform(train_cat)
y_test = encoder.fit_transform(test_cat)

num_classes = np.max(y_train) + 1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

batch_size = 128
epochs = 2
drop_ratio = 0.1

model = models.Sequential()
model.add(layers.Dense(300, input_shape=(max_words,)))
model.add(layers.Activation('relu'))

model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

# Evaluate the accuracy of our trained model
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


text_labels = encoder.classes_ 

# Save the model


with open('tokenize_vect.pickle', 'wb') as a_handle:
    pickle.dump(tokenize, a_handle)

with open('text_labels_encoder.pickle', 'wb') as b_handle:
    pickle.dump(text_labels, b_handle)

model.save("model_file.pkl")

