import matplotlib.pyplot as pp
import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error
import pandas
from random import randint




def train_split(data, size):
    used_idxs = []
    X_values = {'dc': [], 'temp': [], 'wind': []}
    y_values = []
    for i in range(size):
        rnd_idx = randint(0, data.size / data.shape[1] - 337)

        if rnd_idx in used_idxs:
            continue
        else:
            used_idxs.append(rnd_idx)

        X_values['dc'].extend(data['dc'][rnd_idx:rnd_idx + 336].tolist())
        X_values['temp'].extend(data['temp'][rnd_idx:rnd_idx + 336].tolist())
        X_values['wind'].extend(data['wind'][rnd_idx:rnd_idx + 336].tolist())
        y_values.append(data['dc'][rnd_idx + 337].tolist())

    return pandas.DataFrame.from_dict(X_values), pandas.DataFrame.from_dict({'dc': y_values})


def input_query(json_str, idx=0):
    tmp_df = pandas.read_json(json_str)

    return pandas.DataFrame.from_dict(
        {'dc': tmp_df['dc'][idx],
         'temp': tmp_df['temp'][idx],
         'wind': tmp_df['wind'][idx]}
    )


def input_result(json_str, idx=0):
    tmp_df = pandas.read_json(json_str)

    return tmp_df.values[idx]


def train(nn, X_train, y_train, X_val, y_val, steps=100):
    evaluation = []
    for i in range(steps):
        nn.train(X_train, y_train, batch_size=int(len(X_train['dc'].tolist())/336), steps=100)
        evaluation.append(nn.evaluate(X_val, y_val))
        print("Training %s of %s" % ((i+1), steps))
    return evaluation


def plot_training(evaluation):
    loss = []
    for e in evaluation:
        loss.append(e['average_loss'])

    pp.plot(loss)
    # Needed for execution in PyCharm
    pp.show()


def predict(nn, X_pred):
    pred = nn.predict1h(X_pred)
    # Cap results to 0
    predictions = np.array([max(p['predictions'], 0) for p in pred])
    return predictions


def predict24h(nn, X_pred):
    predictions = []

    input = {'dc': X_pred['dc'].tolist()}

    for i in range(24):
        pred = nn.predict1h(pandas.DataFrame.from_dict(input))
        # Cap prediction to 0
        predictions.extend(list([max(p['predictions'][0], 0) for p in pred]))
        # Remove first value and append predicted value
        del input['dc'][0]
        input['dc'].append(predictions[-1])
        print("Prediction for hour %d/%d" % (i+1, 24))

    return predictions


def eval_prediction(prediction, result):
    print("The Explained Variance: %.2f" % explained_variance_score(
        result, prediction))
    print("The Mean Absolute Error: %.2f volt dc" % mean_absolute_error(
        result, prediction))
    print("The Median Absolute Error: %.2f volt dc" % median_absolute_error(
        result, prediction))

