import numpy as np
import matplotlib.pyplot as plt
import cv2 
from sklearn.datasets import make_moons

X, y = make_moons(1000, noise=.05)
Xrnd = np.round(X, decimals = 2)*100

xmin = np.min(Xrnd[:, 0])
xmax = np.max(Xrnd[:, 0])
ymin = np.min(Xrnd[:, 1])
ymax = np.max(Xrnd[:, 1])

plt.ylim((ymin, ymax))
plt.xlim((xmin, xmax))
plt.scatter(Xrnd[:,0], Xrnd[:,1], s=40, c=y, cmap=plt.cm.Spectral)

img = np.zeros((int(ymax-ymin)+1, int(xmax - xmin) + 1))
for i in range(len(Xrnd)):
    x = int(Xrnd[i][0] - xmin)
    z = int(Xrnd[i][1] - ymin)
    if y[i] == 1:
        img[z][x] = 255
    else:
        img[z][x] = 255

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

imgCopy = np.uint8(img)
edges = cv2.Canny(imgCopy, np.min(imgCopy), np.max(imgCopy))

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()