import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pp
import pywatts.neural
from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error
import pandas
from random import randint

from sklearn.model_selection import train_test_split


df = pywatts.db.rows_to_df(list(range(1, 50)))
X = df
y = df['dc']

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=34)

X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)

feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]
n = pywatts.neural.Net(feature_cols=feature_cols)


def train_split(data, size):
    X_values = {'dc': [], 'temp': [], 'wind': []}
    y_values = []
    for i in range(size):
        rnd_idx = randint(0, data.size / data.shape[1] - 337)

        X_values['dc'].extend(data['dc'][rnd_idx:rnd_idx + 336])
        X_values['temp'].extend(data['temp'][rnd_idx:rnd_idx + 336])
        X_values['wind'].extend(data['wind'][rnd_idx:rnd_idx + 336])
        y_values.append(data['dc'][rnd_idx + 337])

    return pandas.DataFrame.from_dict(X_values), pandas.DataFrame.from_dict({'dc': y_values})


def input_data(json_str, idx=0):
    tmp_df = pandas.read_json(json_str)

    return pandas.DataFrame.from_dict(
        {'dc': tmp_df['dc'][idx],
         'temp': tmp_df['temp'][idx],
         'wind': tmp_df['wind'][idx]}
    )


def train(steps=100):
    evaluation = []
    for i in range(steps):
        n.train(X_train, y_train, steps=100)
        evaluation.append(n.evaluate(X_val, y_val))
        print("Training %s of %s" % ((i+1), steps))
    return evaluation


def plot_training(evaluation):
    loss = []
    for e in evaluation:
        loss.append(e['average_loss'])
    pp.plot(loss)


def predict(X_pred):
    pred = n.predict1h(X_pred)
    predictions = np.array([p['predictions'][0] for p in pred])
    return predictions


def eval_prediction(prediction, result):
    print("The Explained Variance: %.2f" % explained_variance_score(
        result, prediction))
    print("The Mean Absolute Error: %.2f volt dc" % mean_absolute_error(
        result, prediction))
    print("The Median Absolute Error: %.2f volt dc" % median_absolute_error(
        result, prediction))

