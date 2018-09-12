import os
import sys

import tensorflow as tf

import pywatts.db
from pywatts.routines import *

# get rid of TF debug message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if len(sys.argv) != 2:
    print("Usage: python photovoltaic_gruppe4.py <file.json>")
    exit(1)

json_file = sys.argv[1]  # json file

oneH, queries = input_queries(json_file)

feature_col = [tf.feature_column.numeric_column(str(idx)) for idx in range(336)]
n = pywatts.neural.Net(feature_cols=feature_col)

predictions = []
for query in queries:
    if oneH:
        predictions.extend(predict(n, query).astype('Float64').tolist())
    else:
        predictions.append(predict24h(n, query))
print(predictions)
