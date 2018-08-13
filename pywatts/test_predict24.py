import tensorflow as tf
import pywatts.db
from pywatts.main import *
import matplotlib.pyplot as pp


PREDICT_QUERY = "query-sample_24hour.json"
PREDICT_RESULT = PREDICT_QUERY.replace("query", "result")
QUERY_ID = 0


pred_query = input_query("../sample_data/" + PREDICT_QUERY, QUERY_ID)
pred_result = input_result("../sample_data/" + PREDICT_RESULT, QUERY_ID)


# Define feature columns and initialize Regressor
feature_col = [tf.feature_column.numeric_column(str(idx)) for idx in range(336)]
n = pywatts.neural.Net(feature_cols=feature_col)

prediction = predict24h(n, pred_query)

print(prediction)
print(pred_result)

pp.plot(pred_result, 'black')
pp.plot(prediction, 'red')
pp.show()

#pywatts.main.eval_prediction(prediction, pred_result)
