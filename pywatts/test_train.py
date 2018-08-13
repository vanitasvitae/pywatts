import peewee
import tensorflow as tf
import pywatts.db
from pywatts.routines import *

NUM_STATIONS_FROM_DB = 75
NUM_TRAIN_STATIONS = 400
NUM_EVAL_STATIONS = 40
TRAIN = True
PLOT = True
TRAIN_STEPS = 50


df = pywatts.db.rows_to_df(list(range(1, NUM_STATIONS_FROM_DB)))
X = df
y = df['dc']

#X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=34)
#X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)


# Define feature columns and initialize Regressor
feature_col = [tf.feature_column.numeric_column(str(idx)) for idx in range(336)]
n = pywatts.neural.Net(feature_cols=feature_col)


# Training data
(X_train, y_train) = train_split(df, NUM_TRAIN_STATIONS)

# Evaluation data
(X_val, y_val) = train_split(df, NUM_EVAL_STATIONS)



train_eval = {}

if TRAIN:

    # Train the model with the steps given
    train_eval = train(n, X_train, y_train, X_val, y_val, TRAIN_STEPS)



if PLOT:
    # Plot training success rate (with 'average loss')
    pywatts.routines.plot_training(train_eval)


exit()
