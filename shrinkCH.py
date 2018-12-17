import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from scipy.spatial import ConvexHull
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from mpl_toolkits import mplot3d
from sklearn.svm import SVC
import random
from mlxtend.plotting import plot_decision_regions

#X, y = make_moons(1000, noise=.05)
X, y = make_blobs(n_samples=10000, centers=2, cluster_std = 1.5, n_features=2, random_state=0)

# First calculate convex hulls of the data set
X0 = X[y == 1]
X0rand = np.copy(X0)
random.shuffle(X0rand)
X1 = X[y==0]
X1rand = np.copy(X1)
random.shuffle(X1rand)
index = int(.8*len(X0))
X0train = X0rand[:index, :]
X1train = X1rand[:index, :]
Xtrain = np.vstack((X0train, X1train))
Xtest = np.vstack((X0rand[index:, :], X1rand[index:, :]))
ytrain = np.append([1]*index, [0]*index)
ytest = np.append([1] *(len(X0rand) - index),[0] *(len(X1rand) - index) )
hull0 = ConvexHull(X0train)
hull1 = ConvexHull(X1train)
h0X = hull0.points[hull0.vertices,:]
h1X = hull1.points[hull1.vertices,:]

# Get centroid
def centroid(data):
    center = []
    for i in range(len(data[0])):
        center.append(np.mean(data[:, i]))
    return center
    
    
# Shrink the convex hull
def shrinkVert(chX, mu):
    #  calc centroid, then find polar coordinates
    center = centroid(chX)
    if len(center) == 2:
        polarCoord = polarTransform2D(chX)
        polarCoord[:, 0] *= mu
        shrX = cartTransform(polarCoord, center)
        # return shruken vertices
        return shrX
    else:
        polarCoord = polarTransform3D(chX)
        polarCoord[:, 0] *= mu
        shrX = cartTransform(polarCoord, center)
        # return shruken vertices
        return shrX
    
def polar2D(x,y):
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    return r,theta

def cartTransform(polX, center):
    Xcart = []
    if len(polX[0]) == 2:
        for pt in polX:
            Xcart.append(cart2D(pt[0], pt[1], center))
    else:
        for pt in polX:
            Xcart.append(cart3D(pt[0], pt[1], pt[2], center))
    return np.asarray(Xcart)
    
def cart2D(r, theta, center):
    x = r*np.cos(theta)+ center[0]
    y = r*np.sin(theta)+ center[1]
    return x,y

def polarTransform2D(X_cart):
    x_centroid = np.mean(X_cart[:,0])
    y_centroid = np.mean(X_cart[:,1])
    X_trans = X_cart - [x_centroid, y_centroid]
    X_res = [polar2D(x[0],x[1]) for x in X_trans]
    return np.asarray(X_res)

def cart3D(r, theta, phi, center):
    x = r*np.sin(theta)*np.cos(phi) + center[0]
    y = r*np.sin(theta)*np.sin(phi) + center[1]
    z = r*np.cos(theta) + center[2]
    return x,y,z

def polar3D(x,y,z):
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arctan2(np.sqrt(x**2+y**2),z)
    phi = np.arctan2(y,x)
    return r,theta,phi

def polarTransform3D(X_cart):
    x_centroid = np.mean(X_cart[:,0])
    y_centroid = np.mean(X_cart[:,1])
    z_centroid = np.mean(X_cart[:,2])
    X_trans = X_cart - [x_centroid, y_centroid, z_centroid]
    X_res = [polar3D(x[0],x[1],x[2]) for x in X_trans]
    return np.asarray(X_res)

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` is a point
    `hull` is either a Delauney or ConvexHull object
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull.points[hull.vertices])

    return hull.find_simplex(p)>=0

# Keep on reducing the shrinked convex hull until they don't intersect
def shrinkCHull(hull0, hull1):
    lRate = .95
    vert0 = hull0.points[hull0.vertices,:]
    vert1 = hull1.points[hull1.vertices,:]
    while True:
        if not sum(in_hull(vert0, hull1)) > 0 and not sum(in_hull(vert1, hull0))>0:
            break
        vert0  = shrinkVert(vert0, lRate)
        vert1 = shrinkVert(vert1, lRate)
        hull0 = ConvexHull(vert0)
        hull1 = ConvexHull(vert1)
    return hull0, hull1

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

sh0, sh1 = shrinkCHull(hull0, hull1)
sh0X = sh0.points[sh0.vertices,:]
sh1X = sh1.points[sh1.vertices,:]
shTrain = np.vstack((sh0X,sh1X))
yShTr = np.append([1]*len(sh0X), [0]*len(sh1X))

'''#%matplotlib notebook
ax = plt.axes(projection='3d')
ax.scatter3D(sh0X[:, 0], sh0X[:, 1], sh0X[:, 2], c= 'black');
ax.scatter3D(sh1X[:, 0], sh1X[:, 1], sh1X[:, 2], c= 'r');
plt.show()

#%matplotlib notebook
ax = plt.axes(projection='3d')
ax.scatter3D(h0X[:, 0], h0X[:, 1], h0X[:, 2], c= 'black');
ax.scatter3D(h1X[:, 0], h1X[:, 1], h1X[:, 2], c= 'red');
plt.show()'''


'''clf = SVC(kernel='linear')
clf.fit(Xtrain, ytrain)
pred = clf.predict(Xtest)
error = pred - ytest
print(np.sum(np.abs(error))/len(ytest))

plot_decision_regions(Xtest, ytest, clf )'''

clf = SVC(kernel='linear', random_state = 0)
clf.fit(shTrain, yShTr)
pred = clf.predict(Xtest)
error = pred - ytest
print(np.sum(np.abs(error))/len(ytest))

plot_decision_regions(Xtest, ytest, clf )
#plot_decision_regions(shTrain, yShTr, clf )
plt.show()