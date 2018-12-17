from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

'''X, y = make_blobs(n_samples=1000, centers=2, cluster_std = 1, n_features=2, random_state=0)

def plotX2D(X_given,y_given, markerSize):
    x_vals_0 = [x[0] for i,x in enumerate(X_given) if y_given[i]==0]
    y_vals_0 = [x[1] for i,x in enumerate(X_given) if y_given[i]==0]
    x_vals_1 = [x[0] for i,x in enumerate(X_given) if y_given[i]==1]
    y_vals_1 = [x[1] for i,x in enumerate(X_given) if y_given[i]==1]
    plt.scatter(x_vals_0, y_vals_0, c = 'b', s=markerSize)
    plt.scatter(x_vals_1, y_vals_1, c = 'r', s =markerSize)
    plt.show()

plotX2D(X,y, 10)'''

X_, y_ = make_blobs(n_samples=1000, centers=2, n_features=3, random_state=0)

def plotX3D(X_given, y_given):
    ax = plt.axes(projection='3d')
    x_vals_0 = [x[0] for i,x in enumerate(X_given) if y_given[i]==0]
    y_vals_0 = [x[1] for i,x in enumerate(X_given) if y_given[i]==0]
    z_vals_0 = [x[2] for i,x in enumerate(X_given) if y_given[i]==0]
    x_vals_1 = [x[0] for i,x in enumerate(X_given) if y_given[i]==1]
    y_vals_1 = [x[1] for i,x in enumerate(X_given) if y_given[i]==1]
    z_vals_1 = [x[2] for i,x in enumerate(X_given) if y_given[i]==1]

    ax.scatter3D(x_vals_0, y_vals_0, z_vals_0, c= 'b');
    ax.scatter3D(x_vals_1, y_vals_1, z_vals_1, c= 'r');
    plt.show()

plotX3D(X_,y_)