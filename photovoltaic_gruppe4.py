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
total = len(queries)
for idx, query in enumerate(queries):

    percent = idx / total
    sys.stdout.write("\r")
    progress = ""
    for i in range(20):
        if i < int(20 * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()

    if oneH:
        predictions.extend(predict(n, query).astype('Float64').tolist())
    else:
        predictions.append(predict24h(n, query))

print(predictions, file=open("test_data_gruppe4.json", "w"))

sys.stdout.write("\r")
print("[ ==================== ] 100.00%")
