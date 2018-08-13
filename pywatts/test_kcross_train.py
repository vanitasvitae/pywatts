import peewee
import tensorflow as tf
import pywatts.db
from pywatts import kcross

NUM_STATIONS_FROM_DB = 75
K = 4
NUM_EVAL_STATIONS = 40
TRAIN = True
PLOT = True
TRAIN_STEPS = 4


df = pywatts.db.rows_to_df(list(range(1, NUM_STATIONS_FROM_DB)))
X = df
y = df['dc']


# Define feature columns and initialize Regressor
feature_col = [tf.feature_column.numeric_column(str(idx)) for idx in range(336)]
n = pywatts.neural.Net(feature_cols=feature_col)


# Training data
(X_train, y_train, X_eval, y_eval) = kcross.split(df, K)


#train_eval = {}

if TRAIN:
    # Train the model with the steps given
    train_eval = kcross.train(n, X_train, y_train, X_eval, y_eval, TRAIN_STEPS)



if PLOT:
    # Plot training success rate (with 'average loss')
    pywatts.routines.plot_training(train_eval)


exit()
