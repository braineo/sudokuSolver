import os
import struct
import numpy as np

"""
Parse images from MNIST binary file.
Idea from https://gist.github.com/akesling/5358964
"""


class MNISTdataset(object):

    def __init__(self, pathToDataset='.'):
        self._pathToDataset = pathToDataset
        return

    def read(self, dataset="training"):

        if dataset is "training":
            fname_img = os.path.join(
                self._pathToDataset, 'train-images.idx3-ubyte')
            fname_lbl = os.path.join(
                self._pathToDataset, 'train-labels.idx1-ubyte')
        elif dataset is "testing":
            fname_img = os.path.join(
                self._pathToDataset, 't10k-images.idx3-ubyte')
            fname_lbl = os.path.join(
                self._pathToDataset, 't10k-labels.idx1-ubyte')
        else:
            raise ValueError, "dataset must be 'testing' or 'training'"

        # Load everything in some numpy arrays
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)

        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(
                len(lbl), rows, cols)

        return img, lbl


    def showImage(self, image):
        """
        :param image: numpy.uint8 2D array of pixel data
        :return: None
        """
        from matplotlib import pyplot
        import matplotlib
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        imgplot = ax.imshow(image, cmap=matplotlib.cm.Greys)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()

dataset = MNISTdataset('../training_data/')
image, label = dataset.read('testing')
np.save('imageTest', image)
np.save('labelTest', label)