from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
import time
from keras.models import Sequential
from keras.layers import Dense

#X,y = make_circles(n_samples = 1000, noise = .2, random_state = 37, factor = 0.3)

X, y = make_blobs(n_samples=1000, centers=2, cluster_std = 1, n_features=2, random_state=0)

np.random.seed(1)

model = Sequential()
model.add(Dense(12, input_dim = 2, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X, y, epochs = 150, batch_size = 10   )

scores = model.evaluate(X, y)
# print '\n', model.metrics_names[1], scores[1]*100

predictions = model.predict(X)
rounded = [round(x[0]) for x in predictions]
print(rounded)