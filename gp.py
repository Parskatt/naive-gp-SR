import numpy as np
import matplotlib.pyplot as plt

def kernel(x_i,x_j,sigma=1,ell=1):
    diff = (x_i-x_j)
    sim = sigma**2*np.exp(-np.einsum("ijd,ijd->ij",diff,diff)/(2*ell**2))
    return sim

def upsample(im, calculate_confidence=False,plot=True):
    assert im.shape == (32,32,3)
    Y_grid,X_grid = np.mgrid[0:31:32j,0:31:32j]
    X = np.stack((Y_grid.flatten(),X_grid.flatten()),axis=-1).reshape(1024,1,2)
    K = kernel(X,X.transpose((1,0,2)))
    K = K + 1e-4*np.eye(*K.shape)

    Y_grid,X_grid = np.mgrid[0:31:128j,0:31:128j]
    X_star = np.stack((Y_grid.flatten(),X_grid.flatten()),axis=-1).reshape(1024*16,1,2)
    if calculate_confidence:
        K_star = kernel(X_star,X_star.transpose((1,0,2)))
    K_cross = kernel(X,X_star.transpose((1,0,2)))
    invK = np.linalg.inv(K)

    f = im.reshape(32*32,3)/255.0
    mu = K_cross.T@invK@(f)
    if calculate_confidence:
        covariance = K_star-K_cross.T@invK@K_cross
    if plot:
        plt.figure(1)
        plt.imshow(f.reshape(32,32,3))
        plt.figure(2)
        plt.imshow(mu.reshape(128,128,3))#mu.reshape(64,64))
        plt.show()
