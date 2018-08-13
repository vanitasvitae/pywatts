import tensorflow as tf
import pywatts.db
from pywatts.routines import *
from pywatts import kcross

NUM_STATIONS_FROM_DB = 75
K = 2
NUM_EVAL_STATIONS = 40
TRAIN = True
PLOT = True
TRAIN_STEPS = 1
TOTAL_STEPS = 2
NUM_QUERIES = 1
PREDICT_QUERY = "query-sample_24hour.json"
PREDICT_RESULT = PREDICT_QUERY.replace("query", "result")
FIGURE_OUTPUT_DIR = "../figures/"


df = pywatts.db.rows_to_df(list(range(1, NUM_STATIONS_FROM_DB)))
X = df
y = df['dc']


# Define feature columns and initialize Regressor
feature_col = [tf.feature_column.numeric_column(str(idx)) for idx in range(336)]
n = pywatts.neural.Net(feature_cols=feature_col)


# Training data
(X_train, y_train, X_eval, y_eval) = kcross.split(df, K)


if TRAIN:

    train_eval = None

    color_gradient_base = (0.5, 0, 0)
    color_step_width = (0.5/TOTAL_STEPS, 0, 0)

    for i in range(TOTAL_STEPS):
        # Train the model with the steps given
        train_eval = kcross.train(n, X_train, y_train, X_eval, y_eval, TRAIN_STEPS)

        for q in range(NUM_QUERIES):

            pred_query = input_query("../sample_data/" + PREDICT_QUERY, q)
            pred_result = input_result("../sample_data/" + PREDICT_RESULT, q)

            prediction = predict24h(n, pred_query)

            pp.figure(q)

            if i == 0:
                pp.plot(pred_result, 'black')

            pp.plot(prediction, color=color_gradient_base)

        color_gradient_base = tuple([sum(x) for x in zip(color_gradient_base, color_step_width)])

    for i in range(NUM_QUERIES):
        pp.figure(i)
        pp.savefig(FIGURE_OUTPUT_DIR+'{}.pdf'.format(i), orientation='landscape')

    if PLOT:
        # Plot training success rate (with 'average loss')
        pywatts.routines.plot_training(train_eval)

exit()
