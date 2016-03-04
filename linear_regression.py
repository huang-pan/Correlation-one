import datetime

import numpy as np
import csv
from sklearn import linear_model
import itertools
import matplotlib.pyplot as plt

X_training=[]
y_training = []
X_test = []
with open('./data/stock_returns_base150.csv', 'rt') as file:
    reader = csv.reader(file, delimiter=',', skipinitialspace=True)
    reader.next()
    for line in itertools.islice(reader, 0, 50):
        X_training.append([line[2], line[3], line[4]])
        y_training.append(line[1])
    for line in itertools.islice(reader, 0, 50):
        X_test.append([line[2], line[3], line[4]])
num_examples = int(len(X_training)) # training set size

X = np.asarray(X_training).astype(float)
y = np.array(y_training).astype(float)
X_test = np.asarray(X_test).astype(float)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X, y)

# The coefficients
print('Coefficients: \n', regr.coef_)
print (len(regr.predict(X_test)))
for i in range(len(regr.predict(X_test))):
    print (float(regr.predict(X_test[i])))