import numpy as np
from sklearn.model_selection import TimeSeriesSplit



def question():
  print("where are the nuclear wessels?")

def model():
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4, 5, 6])
    tscv = TimeSeriesSplit(n_splits=3)
    print(tscv)
    for train, test in tscv.split(X):
        print("%s %s" % (train, test))
