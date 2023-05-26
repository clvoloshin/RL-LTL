import matplotlib.pyplot as plt
import functools
import numpy as np

def plotlive(func):
    plt.ion()

    @functools.wraps(func)
    def new_func(*args, **kwargs):

        # Clear all axes in the current figure.
        axes = plt.gcf().get_axes()
        for axis in axes:
            axis.cla()

        # Call func to plot something
        result = func(*args, **kwargs)

        # Draw the plot
        plt.draw()
        plt.pause(0.01)

        return result

    return new_func 

def moving_average(a, n=3):
    if len(a) < n: return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

@plotlive
def plot_something_live(axes, xs, ys, title=None):
    for i, (ax, x, y) in enumerate(zip(axes, xs, ys)):
        ax.plot(x, y, '-b')
        ax.plot(x, moving_average(y, 20), '-r')
        if (i ==0) and title:
            ax.set_title(title)
    # ax.set_ylim([0, 100])