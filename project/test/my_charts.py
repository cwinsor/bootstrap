
# Common charting infrastructure

import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(x, ylabel, xlabel, title):
    
    # the histogram of the data
    n, bins, patches = plt.hist(x, facecolor='g')
    ymin = 0
    ymax = max(n) * 1.1
    xmin = min(bins) * 0.9
    xmax = max(bins) * 1.1

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # plt.text(60, .025, r'foobar')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.grid(True)
    plt.show()

