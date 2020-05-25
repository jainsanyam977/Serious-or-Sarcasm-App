from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

import pandas as pd
import numpy as np


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    loaded_model = tf.keras.models.load_model('final_model')

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    max_length = 100
    trunc_type='post'
    padding_type='post'

    if request.method == 'POST':
        message = request.form['message']
        #input
        message = [message]
        #tokenizing and padding
        message = tokenizer.texts_to_sequences(message)
        message = pad_sequences(message, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        #conversion to array
        message = np.array(message)
        #conversion to tf.float32
        message = tf.convert_to_tensor(message, dtype=tf.float32)
        #prediction
        my_prediction = loaded_model.predict(message)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)
