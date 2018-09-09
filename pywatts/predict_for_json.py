import os
import sys

import tensorflow as tf

import pywatts.db
from pywatts.routines import *

# get rid of TF debug message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if len(sys.argv) != 3:
    print("Usage: python predict_for_json.py 24h|1h <file.json>")
    exit(1)

type = sys.argv[1]  # '1h' or '24h'
json_file = sys.argv[2]  # json file

queries = input_queries(json_file)

feature_col = [tf.feature_column.numeric_column(str(idx)) for idx in range(336)]
n = pywatts.neural.Net(feature_cols=feature_col)

predictions = []
for query in queries:
    if type == '1h':
        predictions.extend(predict(n, query).astype('Float64').tolist())
    else:
        predictions.append(predict24h(n, query))
print(predictions)
