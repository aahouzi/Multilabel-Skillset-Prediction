############################################################################################
#                                  Author: Anas AHOUZI                                     #
#                               File Name: app/app.py                                      #
#                           Creation Date: November 17, 2020                               #
#                         Source Language: Python                                          #
#         Repository: https://github.com/aahouzi/Multilabel-Skillset-Prediction.git        #
#                              --- Code Description ---                                    #
#                       Flask API code for the model deployment                            #
############################################################################################


################################################################################
#                                   Packages                                   #
################################################################################
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Input, Embedding, GRU
from tensorflow.keras.models import Model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pickle import load
import pandas as pd
import numpy as np
import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')


################################################################################
#                                Main Code                                     #
################################################################################

app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config['DEBUG'] = True


def get_skills(proba, col_list):
    """
    Returns a list of predicted skills base on the outcome
    probability of each skill.
    :param proba: A list containing the probability for each skill in the dataset.
    :param col_list: A list of all available skills or labels.
    :return:
    """
    n = len(proba)
    skills = [col_list[i] for i in range(n) if proba[i][0][0] >= 0.5]
    return skills, len(skills)


def text_preprocessor(df):
    """
    Apply basic text pre-processing steps (Removing stop words, punctuation,
    stemming) on a pandas column dataframe.
    :param df: A pandas column dataframe.
    :return: A processed dataframe.
    """
    # List of Stop words from nltk library
    nltk_stop_words = stopwords.words('english')
    porter = PorterStemmer()
    stem_words = np.vectorize(porter.stem)

    # Removing punctuation
    df = df.str.replace('[^\w\s]', '').apply(word_tokenize)
    # Removing common english stop words
    df = df.apply(lambda x: [word for word in x if word not in nltk_stop_words])
    # Stemming
    df = df.apply(lambda x: [stem_words(word) for word in x])
    # Join the tokens back to a sentence
    df = df.apply(lambda x: ' '.join(map(str, x)))

    return df


def get_model(tokenizer_obj, n):
    """
    Constructs & returns the base model used for predicting the skill sets.
    :param tokenizer_obj: A tf.Keras tokenizer object.
    :param n: Number of labels to predict.
    :return: A tf.keras functional API model.
    """
    # Size of the vocabulary
    vocab_size = len(tokenizer_obj.word_index) + 1
    # Dimension after padding sequences
    max_len = 100

    # Define the main neural network architecture
    x_in = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, input_length=max_len, output_dim=128)(x_in)
    x = GRU(180)(x)
    dense_layers = [Dense(1, activation='sigmoid', name='Dense{}'.format(_))(x) for _ in range(n)]

    return Model(inputs=x_in, outputs=dense_layers)


def predict(text):
    """
    Loads the tokenizer & the model, and returns the predicted tags.
    :param text: Job description provided by the end user.
    :return: Skill tags corresponding to the job description.
    """

    # Loading our tokenizer and label names
    tokenizer = load(open("app/models/tokenizer.pkl", 'rb'))
    labels_list = pd.read_csv("app/models/labels.csv").columns

    # Construct the model, and load the saved weights
    model = get_model(tokenizer, len(labels_list))
    model.load_weights("app/models/model_weights.h5")

    # Cleaning the job description
    text_series = pd.Series(text)
    clean_text = text_preprocessor(text_series)

    # Converting the cleaned text to sequences for GRU inference
    x_inf = tokenizer.texts_to_sequences(clean_text)
    x_inf = sequence.pad_sequences(x_inf, maxlen=100)

    # Generating prediction probabilities for every label
    proba = model.predict(x_inf.reshape(1, -1))

    # Returning skill tags seperated by a comma, and their number
    skills, m = get_skills(proba, labels_list)

    return skills, m


@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    job_descrip = request.form['job_descrip']
    skills, m = predict(job_descrip)

    return render_template('child.html', skills=skills, m=m)


if __name__ == "__main__":
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
