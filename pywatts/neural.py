import pandas
import numpy as np
import tensorflow as tf


def pywatts_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=1):
    # Create dictionary for features in hour 0 ... 335
    features = {str(idx): [] for idx in range(336)}
    #dc_values = X['dc'].tolist()
    dc_values = X['dc']

    # Iterate the empty dictionary always adding the idx-th element from the dc_values list
    for idx, value_list in features.items():
        value_list.extend(dc_values[int(idx)::336])

    labels = None
    if y is not None:
        #labels = y['dc'].values
        labels = y['dc']

    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(dict(features))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if shuffle:
        dataset.shuffle(len(features['0']))

    return dataset.batch(batch_size)


class Net:
    __regressor = None
    __feature_cols = [tf.feature_column.numeric_column(col) for col in ['dc', 'temp', 'wind']]

    def __init__(self, feature_cols=__feature_cols):
        self.__regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                                     hidden_units=[75, 75],
                                                     model_dir='tf_pywatts_model')

    def train(self, training_data, training_results, batch_size, steps):
        self.__regressor.train(input_fn=lambda: pywatts_input_fn(training_data, y=training_results, num_epochs=None, shuffle=True, batch_size=batch_size), steps=steps)

    def evaluate(self, eval_data, eval_results, batch_size=1):
        return self.__regressor.evaluate(input_fn=lambda: pywatts_input_fn(eval_data, y=eval_results, num_epochs=1, shuffle=False, batch_size=batch_size), steps=1)

    def predict1h(self, predict_data):
        return self.__regressor.predict(input_fn=lambda: pywatts_input_fn(predict_data, num_epochs=1, shuffle=False))
