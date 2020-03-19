import gp
from cifar10 import unpickle
import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) > 1:
    img = plt.imread(sys.argv)
    gp.upsample(img)
else:
    data = unpickle("sample_images/data_batch_1")[b"data"]
    rand_ind = np.random.permutation(len(data))
    data = data[rand_ind]
    for idx,img in enumerate(data):
        img = img.reshape((3,32,32)).transpose((1,2,0))
        plt.figure()
        plt.imshow(img)
        plt.title("Original image (32x32)")
        gp.upsample(img,kernel_name="matern52")
        gp.upsample(img,kernel_name="RBF")
        plt.show()
