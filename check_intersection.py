from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import numpy as np

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` is a point
    `hull` is either a Delauney or ConvexHull object
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull.points[hull.vertices])

    return hull.find_simplex(p)>=0

def polar2D(x,y):
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    return r,theta

def cart2D(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y

def polarTransform2D(X_cart):
    x_centroid = np.mean(X_cart[:,0])
    y_centroid = np.mean(X_cart[:,1])
    X_trans = X_cart - [x_centroid, y_centroid]
    X_res = [polar2D(x[0],x[1]) for x in X_trans]
    return np.asarray(X_res)

def cart3D(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
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
