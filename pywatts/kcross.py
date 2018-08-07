import random
import itertools
from pywatts import db


def split(data, k):
    """Returns (X_train, y_train, X_eval, y_eval)"""

    # Training features as list of dictionaries (each dict is for ONE test run)
    X_train = []
    # Training labels as list of dictionaries (each dict is for ONE test run)
    y_train = []
    # Evaluation features as list of dictionaries (each i-th dict includes all features except X_train[i])
    X_eval = []
    # Evaluation labels as list of dictionaries (each i-th dict includes all labels except X_train[i])
    y_eval = []

    data_list = data['dc'].tolist()

    # Each sample has 337 elements
    samples = [data_list[i:i+337] for i in range(0, len(data_list) - 337, 337)]
    # Randomly shuffle samples
    random.shuffle(samples)

    bucketsize = int(len(samples) / k)

    print(k)
    print(len(data))
    print(len(samples))
    print(bucketsize)

    # K steps
    for i in range(k):
        eval_dict = []
        train_dict = []
        eval_samples = []
        train_samples = []
        for j in range(k):
            if j == i:
                eval_samples.extend(samples[i*bucketsize:(i+1)*bucketsize])
            else:
                train_samples.extend(samples[i*bucketsize:(i+1)*bucketsize])

        for s in eval_samples:
            # Create new dictionaries in the eval lists
            X_eval.append({'dc': s[:-1]})
            y_eval.append({'dc': s[-1]})

        for s in train_samples:
            X_train.append({'dc': s[:-1]})
            y_train.append({'dc': s[-1]})

        print(len(X_train) / 12)
        #print(X_train)
        #print(y_train)
        exit(0)

    return X_train, y_train, X_eval, y_eval


def train(nn, X_train, y_train, X_eval, y_eval, steps=10):
    """Trains the Network nn using k-cross-validation"""
    evaluation = []
    for count, train_data in enumerate(X_train):
        for i in range(steps):
            nn.train(train_data, y_train[count], batch_size=int(len(train_data['dc'])/336), steps=1)
            print(X_eval[count])
            print(len(X_eval[count]['dc']))
            print(y_eval[count])
            evaluation.append(nn.evaluate(X_eval[count], y_eval[count], batch_size=int(len(X_eval[count]['dc'])/336)))
            print("Training %s: %s/%s" % (count, (i+1), steps))





