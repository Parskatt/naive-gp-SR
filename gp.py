import numpy as np
import matplotlib.pyplot as plt

def RBF(x_i,x_j,sigma=1,ell=1):
    diff = (x_i-x_j)
    sim = sigma**2*np.exp(-np.einsum("ijd,ijd->ij",diff,diff)/(2*ell**2))
    return sim

def matern52(x_i,x_j,sigma=1,rho=1):
    dist = np.sqrt(np.sum((x_i-x_j)**2,axis=2))
    sim = sigma*(1+np.sqrt(5)*dist/rho+5*dist**2/(3*rho**2))*np.exp(-np.sqrt(5)*dist/rho)
    return sim

def kernel(x_i,x_j,kernel_name="matern52",**kwargs):
    if kernel_name == "matern52":
        return matern52(x_i,x_j,**kwargs)
    else:
        return RBF(x_i,x_j,**kwargs)

def upsample(im, calculate_confidence=False,plot=True,kernel_name="matern52"):
    assert im.shape == (32,32,3)
    Y_grid,X_grid = np.mgrid[0:31:32j,0:31:32j]
    X = np.stack((Y_grid.flatten(),X_grid.flatten()),axis=-1).reshape(1024,1,2)
    K = kernel(X,X.transpose((1,0,2)),kernel_name=kernel_name)
    K = K + 1e-4*np.eye(*K.shape)

    Y_grid,X_grid = np.mgrid[0:31:128j,0:31:128j]
    X_star = np.stack((Y_grid.flatten(),X_grid.flatten()),axis=-1).reshape(1024*16,1,2)
    if calculate_confidence:
        K_star = kernel(X_star,X_star.transpose((1,0,2)),kernel_name=kernel_name)
    K_cross = kernel(X,X_star.transpose((1,0,2)),kernel_name=kernel_name)
    invK = np.linalg.inv(K)

    f = im.reshape(32*32,3)/255.0
    mu = np.clip(K_cross.T@invK@(f),0,1)
    if calculate_confidence:
        covariance = K_star-K_cross.T@invK@K_cross
    if plot:
        plt.figure()
        plt.imshow(mu.reshape(128,128,3))
        plt.title(f"Upsampled GP image (128x128), using a {kernel_name} kernel")