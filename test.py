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

#loading dataset

with open('tokenize_vect.pickle', 'rb') as a_handle:
    tokenize = pickle.load(a_handle)

with open('text_labels_encoder.pickle', 'rb') as b_handle:
    text_labels = pickle.load(b_handle)

model = tf.keras.models.load_model("model_file.pkl")

app = Flask(__name__)

@app.route('/', methods=['GET'])
def Index():
    return "Hello"


# defining a route
@app.route("/category", methods=['POST'])
def category(): 
    if request.method == "POST":
        html = request.form['desc']
        listx = []
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True
        text = text_maker.handle(html)
        listx.append(text)
        tuple_data = tuple(listx)
        dataset = pd.DataFrame(tuple_data, columns=["description"])
        new_x_test = tokenize.texts_to_matrix(dataset['description'])
        prediction = model.predict(np.array([new_x_test[0]]))
        predicted_label = text_labels[np.argmax(prediction)]
        print(dataset['description'].iloc[0][:200], "...")
        print(predicted_label)
        return predicted_label
    else:
        return "Error"

if __name__ == "__main__":
    app.run(host = '127.0.0.1',port='2012')

