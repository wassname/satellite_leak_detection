from matplotlib import pyplot as plt


# TODO show band labels?
def imshow_bands(X_img, figsize=(14, 14), interpolation=None, labels=None):
    """show mosiac of input bands."""
    plt.figure(figsize=figsize)

    f, axarr = plt.subplots(4, 4)
    for j in range(X_img.shape[0]):
        plt.subplot(4, 4, j + 1)
    #     plt.subplot(j)
    #     ax=axarr.flatten()[i]
        plt.imshow(X_img[j, :, :], interpolation=interpolation)
        plt.grid(False)
        plt.gca().set_axis_off()
        if labels:
            label = labels[j]
        else:
            label = ''
        plt.title('#%i - %s' % (j + 1, label))
    #     plt.show()

    plt.subplot(4, 4, j + 2)
    plt.grid(False)
    plt.gca().set_axis_off()
    plt.subplot(4, 4, j + 3)
    plt.grid(False)
    plt.gca().set_axis_off()
    plt.suptitle('Image bands')
    return plt.gca()
