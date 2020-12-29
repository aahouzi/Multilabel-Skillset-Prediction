############################################################################################
#                                  Author: Anas AHOUZI                                     #
#                               File Name: Train/train_and_evaluate.py                     #
#                           Creation Date: November 17, 2020                               #
#                         Source Language: Python                                          #
#         Repository: https://github.com/aahouzi/Multilabel-Skillset-Prediction.git        #
#                              --- Code Description ---                                    #
#           Training & evaluating a GRU architecture for predicting skill sets             #
############################################################################################


################################################################################
#                                   Packages                                   #
################################################################################
from tensorflow.keras.layers import Dense, Input, Embedding, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from Features.features import get_features
from pickle import load
import numpy as np
import argparse

################################################################################
#                                Main arguments                                #
################################################################################

parser = argparse.ArgumentParser(description='Train a GRU model for predicting skill sets')

parser.add_argument('--n_epochs', type=int, required=True, help='Number of epochs')
parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')

args = parser.parse_args()

################################################################################
#                             Main Functions                                   #
################################################################################


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


################################################################################
#                                Main Code                                     #
################################################################################

# Pre-process the dataset, and load the tokenizer
x_train, x_test, y_train, y_test, labels = get_features('dataset/dataset_finale.csv', save=1)
tokenizer = load(open("app/models/tokenizer.pkl", 'rb'))
n = len(labels)

# Construct & compile the model
model = get_model(tokenizer, n)
model.compile(loss='binary_crossentropy', optimizer=Adam(args.learning_rate), metrics=['accuracy'])

# A list of lists containing the label for every sample in the dataset.
y_train_list = [y_train[category].values for category in labels]
y_test_list = [y_test[category].values for category in labels]

# Train the model.

print('\n\n'+'---' * 10 + ' Training the model ' + '---' * 10)

early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
model.fit(x=x_train, y=y_train_list, batch_size=args.batch_size,
          epochs=args.n_epochs, verbose=1, validation_split=0.1, callbacks=early_stop)

# In Multi Label classification, the evaluate method returns an array containing
# as a first element the total loss, followed by the loss for every output dense
# layer, and then followed by the accuracy for every output dense layer.

# PS: In calculating the test accuracy, I chose the median since there's a huge disparity
# between the accuracies of each output dense layer.

print('\n\n'+'---' * 10 + ' Evaluating the model ' + '---' * 10)

score = model.evaluate(x=x_test, y=y_test_list, verbose=1)
print("\n[INFO]: Total test loss: {}\n".format(round(score[0], 2)))
print("\n[INFO]: Average test accuracy: {}\n".format(round(np.median(score[n+1:])*100, 2)))

# Save weights
model.save_weights('app/models/model_weights.h5')










