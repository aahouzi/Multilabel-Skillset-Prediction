############################################################################################
#                                  Author: Anas AHOUZI                                     #
#                               File Name: Features/features.py                            #
#                           Creation Date: November 17, 2020                               #
#                         Source Language: Python                                          #
#         Repository: https://github.com/aahouzi/Multilabel-Skillset-Prediction.git        #
#                              --- Code Description ---                                    #
#      Implementation of various functions for text pre-processing & feature encoding      #
############################################################################################


################################################################################
#                                   Packages                                   #
################################################################################
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pickle import dump
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')


################################################################################
#                                  Main Code                                   #
################################################################################
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


def get_features(path_to_dataset, save):
    """
    Encoding train/test sets for neural network training.
    :param path_to_dataset: The path to the dataset.
    :param save: 1 to save the tokenizer locally, 0 otherwise.
    :return: Encoded train/test sets.
    """
    # Open the dataset.
    df = pd.read_csv(path_to_dataset)

    # Define our labels vector.
    labels = [col for col in df if col not in ['O*NET-SOC Code', 'Description']]

    # Split the dataset into train and test sets.
    x = df["Description"]
    y = df[labels]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Clean the text, and perform all the usual text pre-processing steps. (Stop Words, Tokenizing, Stemming)
    x_train, x_test = text_preprocessor(x_train), text_preprocessor(x_test)

    # Converting text to sequences for neural network training, and then padding
    # those sequences in order to have the same length all over the dataset.
    tokenizer = Tokenizer(num_words=500)
    tokenizer.fit_on_texts(x_train)

    # Convert tokenized descriptions to sequences
    x_train, x_test = tokenizer.texts_to_sequences(x_train), tokenizer.texts_to_sequences(x_test)

    # Pad the sequences with zeros to have the same length, which is maxlen.
    x_train, x_test = sequence.pad_sequences(x_train, maxlen=100), sequence.pad_sequences(x_test, maxlen=100)

    # Save the tokenizer for inference in the web app
    if save:
        dump(tokenizer, open('app/models/tokenizer.pkl', 'wb'))

    print('\n[INFO]: Training data shape: {}'.format(x_train.shape))
    print('\n[INFO]: Testing data shape: {}\n'.format(x_test.shape))

    return x_train, x_test, y_train, y_test, labels

