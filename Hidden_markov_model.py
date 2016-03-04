import datetime
import numpy as np
import pylab as pl
from sklearn.hmm import GaussianHMM

###############################################################################
# Downloading the data
X_training=[]
y_training = []
X_test = []
with open('./data/stock_returns_base150.csv', 'rt') as file:
    reader = csv.reader(file, delimiter=',', skipinitialspace=True)
    reader.next()
    #Best feature for this problem is S2,S3 and S4 based on Principal Components Attribute Transformer sttribute search
    for line in itertools.islice(reader, 0, 50):
        X_training.append([line[2], line[3], line[4]])
        y_training.append(line[1])
    for line in itertools.islice(reader, 0, 50):
        X_test.append([line[2], line[3], line[4]])
num_examples = int(len(X_training)) # training set size

X = np.asarray(X_training).astype(float)
y = np.array(y_training).astype(float)
X_test = np.asarray(X_test).astype(float)
# take diff of close value
# this makes len(diff) = len(close_t) - 1
# therefore, others quantity also need to be shifted
# pack diff and volume for training
X = np.column_stack([X])

###############################################################################
# Run Gaussian HMM
print("fitting to HMM and decoding ..."),
n_components = 2

# make an HMM instance and execute fit
model = GaussianHMM(n_components, "diag")
model.fit([X])

# predict the optimal sequence of internal hidden state
hidden_states = model.predict(X_test)
for i in range(0,50):
    print(hidden_states[i]),
print("done\n")

###############################################################################
# print trained parameters and plot
print("Transition matrix")
print (model.transmat_)
print ("")

print ("means and vars of each hidden state")
for i in xrange(n_components):
    print ("%dth hidden state" % i)
    print ("mean = ", model.means_[i])
    print ("var = ", np.diag(model.covars_[i]))
    print ("")
