
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



def plot_box_and_whisker():
    # from https://matplotlib.org/3.1.0/gallery/pyplots/boxplot_demo_pyplot.html#sphx-glr-gallery-pyplots-boxplot-demo-pyplot-py

    import numpy as np
    import matplotlib.pyplot as plt

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # fake up some data
    spread = np.random.rand(50) * 100
    center = np.ones(25) * 50
    flier_high = np.random.rand(10) * 100 + 100
    flier_low = np.random.rand(10) * -100
    data = np.concatenate((spread, center, flier_high, flier_low))

    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(data)