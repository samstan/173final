from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
import time


class Dummy:
    def __init__(self, clf):
        self.c = clf
    
    def predict(self, data):
        return np.asarray([self.c.predict([polar(data[i][0],data[i][1])]) for i in range(data.shape[0])])

def polar(x,y):
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    return r,theta

def cart(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y

def polarTransform(X_cart):
    x_centroid = np.mean(X_cart[:,0])
    y_centroid = np.mean(X_cart[:,1])
    X_trans = X_cart - [x_centroid, y_centroid]
    X_res = [polar(x[0],x[1]) for x in X_trans]
    return np.asarray(X_res)

def plotX(X_given,y_given, markerSize):
    x_vals_0 = [x[0] for i,x in enumerate(X_given) if y_given[i]==0]
    y_vals_0 = [x[1] for i,x in enumerate(X_given) if y_given[i]==0]
    x_vals_1 = [x[0] for i,x in enumerate(X_given) if y_given[i]==1]
    y_vals_1 = [x[1] for i,x in enumerate(X_given) if y_given[i]==1]
    plt.scatter(x_vals_0, y_vals_0, c = 'b', s=markerSize)
    plt.scatter(x_vals_1, y_vals_1, c = 'r', s =markerSize)
    plt.show()

X,y = make_circles(n_samples = 10000, noise = .2, random_state = 37, factor = 0.3)
X_test,y_test = make_circles(n_samples = 2000, noise = .2, random_state = 19, factor = 0.3)
X_ex,y_ex = make_circles(n_samples = 200, noise = .2, random_state = 19, factor = 0.3)

#plotX(X_ex,y_ex, 10)

t1 = time.time()
X_polar = polarTransform(X)

'''r_vals_0 = [x[0] for i,x in enumerate(X_polar) if y[i]==0]
t_vals_0 = [x[1] for i,x in enumerate(X_polar) if y[i]==0]
r_vals_1 = [x[0] for i,x in enumerate(X_polar) if y[i]==1]
t_vals_1 = [x[1] for i,x in enumerate(X_polar) if y[i]==1]
plt.scatter(t_vals_0, r_vals_0, c = 'b')
plt.scatter(t_vals_1, r_vals_1, c = 'r')

plt.show()'''

clf = SVC(kernel = 'linear')
clf.fit(X_polar,y)

'''clf_dummy = Dummy(clf)


#plot decision boundary in polar coordinates
plot_decision_regions(polarTransform(X_ex), y_ex, clf)
plt.show()

#plot decision boundary in Cartesian coordinates
plot_decision_regions(X_ex, y_ex, clf_dummy)
plt.show()'''
t2 = time.time()
def classify_polar(data):
    return clf.predict([polar(data[0],data[1])])

preds = [classify_polar(x)[0] for x in X_test]

counter = 0
for i in range(len(y_test)):
    if preds[i]==y_test[i]:
        counter+=1

print(counter)
print('\n')
print(t2-t1)

t1 = time.time()
clf2 = SVC(kernel = 'rbf', decision_function_shape='ovo')
clf2.fit(X,y)
t2 = time.time()
def classify(data):
    return clf2.predict([data])

preds2 = [classify(x)[0] for x in X_test]


counter = 0
for i in range(len(y_test)):
    if preds2[i]==y_test[i]:
        counter+=1

print(counter)
print('\n')
print(t2-t1)



    

