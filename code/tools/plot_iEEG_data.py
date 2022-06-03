from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.collections import LineCollection
import pandas as pd


def plot_iEEG_data(data, t, linecolor='k'):
    """"
    2021.06.23. Python 3.8
    Akash Pattnaik
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Purpose:
    To plot iEEG data
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Input
        data: iEEG data in pandas.DataFrame or numpy.array
        time: time array
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Output:
        Returns figure handle
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    n_rows = data.shape[1]
    h, w = n_rows * 1.5, 15
    plt.figure(figsize=(w, h))

    ax = plt.axes()

    dmin = np.average(data.min().array)
    dmax = np.average(data.max().array)

    dr = (dmax - dmin)  # Crowd them a bit.

    segs = []
    for i in range(n_rows):
        if isinstance(data, pd.DataFrame):
            segs.append(np.column_stack((t, data.iloc[:, i])))
        elif isinstance(data, np.ndarray):
            segs.append(np.column_stack((t, data[:, i])))
        else:
            print("Data is not in valid format")

    offsets = (np.arange(n_rows) * dr)[::-1]

    lines = np.array(segs)

    mean = np.mean(lines[:, :, 1], axis=1)

    pltlines = lines[:, :, 1] - mean[:, np.newaxis] + offsets[:, np.newaxis]

    plt.plot(lines[:, :, 0].T, pltlines.T, 'k', linewidth=0.2)

    # # Set the yticks to use axes coordinates on the y axis
    plt.yticks(offsets)
    if isinstance(data, pd.DataFrame):
        ax.set_yticklabels(data.columns)
    plt.xlabel('Time (s)')
    plt.title('dr = %f' %dr)

    plt.show()
