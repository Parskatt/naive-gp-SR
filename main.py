import gp
from cifar10 import unpickle
import sys
import matplotlib.pyplot as plt
if len(sys.argv) > 1:
    img = plt.imread(sys.argv)
    gp.upsample(img)
else:
    data = unpickle("sample_images/data_batch_1")[b"data"]
    for img in data:
        img = img.reshape((3,32,32)).transpose((1,2,0))
        gp.upsample(img)