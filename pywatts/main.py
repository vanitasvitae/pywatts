import tensorflow as tf
import pywatts.neural

from sklearn.model_selection import train_test_split

df = pywatts.db.rows_to_df(list(range(1, 50)))
X = df[[col for col in df.columns if col != 'dc']]
y = df['dc']

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)

feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]
n = pywatts.neural.Net(feature_cols=feature_cols)