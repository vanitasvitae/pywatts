import tensorflow as tf
import matplotlib.pyplot as pp
import pywatts.neural
import pandas

from sklearn.model_selection import train_test_split


df = pywatts.db.rows_to_df(list(range(1, 50)))
X = df[[col for col in df.columns if col != 'dc']]
y = df['dc']

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)

X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)

feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]
n = pywatts.neural.Net(feature_cols=feature_cols)


def input_data(json_str):
    tmp_df = pandas.read_json(json_str)

    return pandas.DataFrame.from_dict(
        {'dc': [x for l in tmp_df['dc'] for x in l],
         'temp': [x for l in tmp_df['temp'] for x in l],
         'wind': [x for l in tmp_df['wind'] for x in l]}
    )


def train(steps=100):
    evaluation = []
    for i in range(steps):
        n.train(X_train, y_train, steps=400)
        evaluation.append(n.evaluate(X_val, y_val))
        print("Training %s of %s" % (i, steps))
    return evaluation


def plot_training(evaluation):
    loss = []
    for e in evaluation:
        loss.append(e['loss'])
    pp.plot(loss)

