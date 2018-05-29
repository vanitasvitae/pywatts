import tensorflow as tf


class Net:
    regressor = None


    def net(self, feature_cols):
        self.regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                                   hidden_units=[50, 50],
                                                   model_dir='tf_pywatts_model')

    def pywatts_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):
        return tf.estimator.inputs.pandas_input_fn(x=X,
                                                   y=y,
                                                   num_epochs=num_epochs,
                                                   shuffle=shuffle,
                                                   batch_size=batch_size)

    def train(self, training_data, steps):
        self.regressor.train(input_fn=self.pywatts_input_fn(training_data, num_epochs=None, shuffle=True), steps=steps)

    def evaluate(self, eval_data):
        self.regressor.evaluate(input_fn=self.pywatts_input_fn(eval_data, num_epochs=1, shuffle=False), steps=1)

    def predict1h(self, df):
        df = df.drop(['month', 'day', 'hour'])
        predictions = self.regressor.predict(input_fn=self.pywatts_input_fn(df, num_epochs=1, shuffle=False))



