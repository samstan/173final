from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
import time
from keras.models import Sequential
from keras.layers import Dense
import random

#X,y = make_circles(n_samples = 1000, noise = .2, random_state = 37, factor = 0.3)

X, y = make_blobs(n_samples=1000, centers=2, cluster_std = 1.5, n_features=3, random_state=0)

X0 = X[y == 1]
X0rand = np.copy(X0)
random.shuffle(X0rand)
X1 = X[y==0]
X1rand = np.copy(X1)
random.shuffle(X1rand)
index = int(.8*len(X0))
Xtrain = np.vstack((X0rand[:index, :], X1rand[:index, :]))
Xtest = np.vstack((X0rand[index:, :], X1rand[index:, :]))
ytrain = np.append([1]*index, [0]*index)
ytest = np.append([1] *(len(X0rand) - index),[0] *(len(X1rand) - index) )

np.random.seed(1)

model = Sequential()
model.add(Dense(12, input_dim = 3, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(Xtrain, ytrain, epochs = 100, batch_size = 10   )

#scores = model.evaluate(X, y)
# print '\n', model.metrics_names[1], scores[1]*100

'''predictions = model.predict(X_test)

num_incorrect = 0
for idx, el in predictions:
    if int(np.round(el))!=y[idx]:
        num_incorrect+=1
print()'''

print(model.evaluate(Xtest,ytest))
