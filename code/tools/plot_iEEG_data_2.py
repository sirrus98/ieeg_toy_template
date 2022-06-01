from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.collections import LineCollection
import pandas as pd


def plot_iEEG_data_2(EEG, t, vspace=100, color='k'):
    '''
    Plot the EEG data, stacking the channels horizontally on top of each other.

    Parameters
    ----------
    EEG : array (channels x samples)
        The EEG data
    vspace : float (default 100)
        Amount of vertical space to put between the channels
    color : string (default 'k')
        Color to draw the EEG in
    '''

    n_row = EEG.shape[1]

    bases = vspace * np.arange(n_row - 1, -1, -1)  # vspace * 0, vspace * 1, vspace * 2, ..., vspace * 6

    # To add the bases (a vector of length 7) to the EEG (a 2-D Matrix), we don't use
    # loops, but rely on a NumPy feature called broadcasting:
    # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    test = EEG.T

    data = EEG.T + bases[:, np.newaxis]

    # Calculate a timeline in seconds, knowing that the sample rate of the EEG recorder was 2048 Hz.

    # Plot EEG versus time
    plt.plot(data, color=color)

    # Add gridlines to the plot
    plt.grid()

    # Label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Channels')

    print('here')
    # The y-ticks are set to the locations of the electrodes. The international 10-20 system defines
    # default names for them.
    # plt.gca().yaxis.set_ticks(bases)
    # plt.gca().yaxis.set_ticklabels(['Fz', 'Cz', 'Pz', 'CP1', 'CP3', 'C3', 'C4'])

    # Put a nice title on top of the plot
    plt.title('EEG data')

    plt.show()
