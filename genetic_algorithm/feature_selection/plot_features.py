import numpy as np
import matplotlib.pyplot as plt

def plot_feature_classes(*features):
    fig = plt.figure()
    for feat in features:
        plt.plot(feat[:, 0], feat[:, 1], '.', alpha=0.3)
    plt.show()

def plot_with_colorscale(feat, color_data, alpha=1, colormap=plt.cm.get_cmap("RdBu"), norm=None):
    fig = plt.figure()
    cb = plt.scatter(feat[:, 0], feat[:, 1], alpha=alpha, c=color_data, cmap=colormap, norm=norm)
    plt.colorbar(cb)
    plt.show()