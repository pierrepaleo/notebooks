import numpy as np
import matplotlib.pyplot as plt

def ims(img, cmap=None, legend=None, nocbar=False, share=True):
    """
    image visualization utility.

    img: 2D numpy.ndarray, or list of 2D numpy.ndarray
        image or list of images
    cmap: string
        Optionnal, name of the colorbar to use.
    legend: string, or list of string
        legend under each image
    nocbar: bool
        if True, no colorbar are displayed. Default is False
    share: bool
        if True, the axis are shared between the images, so that zooming in one image
        will zoom in all the corresponding regions of the other images. Default is True
    """
    try:
        _ = img.shape
        nimg = 1
    except AttributeError:
        nimg = len(img)
    #
    if (nimg <= 2): shp = (1,2)
    elif (nimg <= 4): shp = (2,2)
    elif (nimg <= 6): shp = (2,3)
    elif (nimg <= 9): shp = (3,3)
    else: raise ValueError("too many images")
    #
    plt.figure()
    for i in range(nimg):
        curr = list(shp)
        curr.append(i+1)
        curr = tuple(curr)
        if nimg > 1:
            if i == 0: ax0 = plt.subplot(*curr)
            else:
                if share: plt.subplot(*curr, sharex=ax0, sharey=ax0)
                else: plt.subplot(*curr)
            im = img[i]
            if legend: leg = legend[i]
        else:
            im = img
            if legend: leg = legend
        if cmap:
            plt.imshow(im, cmap=cmap, interpolation="nearest")
        else:
            plt.imshow(im, interpolation="nearest")
        if legend: plt.xlabel(leg)
        if nocbar is False: plt.colorbar()

    plt.show()


def gradient(img):
    '''
    Compute the gradient of an image as a numpy array
    Code from https://github.com/emmanuelle/tomo-tv/
    '''
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    slice_all = [0, slice(None, -1),]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient
