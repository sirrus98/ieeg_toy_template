import numpy as np
from scipy.fft import fft
import scipy.signal as sig


# TODO:
#   average frequency-domain magnitude in five frequency bands: 5-15 Hz, 20-25 Hz, 75-115 Hz, 125-160 Hz, 160-175
#   overall power/ overall coherence (but hard to do given the channels we have)

def findInd(fr, fs=1000, pts=0):
    '''
    Calculates the index needed to find the right frequency value
    :param fr: frequency of interest
    :param fs: sampling frequency
    :param pts: number of points in the signal
    :types: float
    :return: idx
    :rtype: float
    '''
    # to get indices, x/len(p) = freq/500 so x = len(p)*freq/500
    return int(fr * pts / (fs / 2))


def freqFeatures(filtered_window, fs=1000):
    """
      Input:
        filtered_window (window_samples x channels): the window of the filtered ecog signal
        fs: sampling rate
      Output:
        features (channels x num_features): the features calculated on each channel for the window
    """
    features = []
    for channel in filtered_window:
        avg_voltage = np.mean(channel, axis=0)

        # Fourier transform, take real component, take absolute value, and convert to
        # decibels
        # pad to 1000 points to get 1 Hz resolution
        p = 20 * np.log10(np.abs(np.fft.rfft(channel, axis=0)) + 0.001)  # create power spectrum
        f = np.linspace(0, fs / 2, len(p))  # create corresponding frequencies

        avPow = np.mean(p, axis=0)  # average power

        alpha = np.mean(p[findInd(8, fs, len(p)):findInd(14, fs, len(p)) + 1], axis=0)  # alpha band
        # mu = np.mean(p[findInd(8):findInd(13)] + 1)  # mu band, apparently tied to movement
        beta = np.mean(p[findInd(14, fs, len(p)):findInd(30, fs, len(p)) + 1], axis=0)  # beta band
        low_gamma = np.mean(p[findInd(30, fs, len(p)):findInd(60, fs, len(p)) + 1], axis=0)  # gamma band ish
        gamma = np.mean(p[findInd(60, fs, len(p)):findInd(100, fs, len(p)) + 1], axis=0)
        high_gamma = np.mean(p[findInd(100, fs, len(p)):findInd(200, fs, len(p)) + 1], axis=0)
        # f1 = np.mean(p[findInd(5, fs, len(p)):findInd(15, fs, len(p)) + 1])  # freq band 1
        # f2 = np.mean(p[findInd(20, fs, len(p)):findInd(25, fs, len(p)) + 1])  # freq band 2
        # f3 = np.mean(p[findInd(75, fs, len(p)):findInd(115, fs, len(p)) + 1])  # freq band 3
        # f4 = np.mean(p[findInd(125, fs, len(p)):findInd(160, fs, len(p)) + 1])  # freq band 4
        # f5 = np.mean(p[findInd(160, fs, len(p)):findInd(175, fs, len(p)) + 1])  # freq band 5

        # From what I found, f1,f2, f3, f4, f5 are best, along with avg_voltage. Included the others just in case.
        features.append([avPow, avg_voltage, alpha, beta, low_gamma, gamma, high_gamma])

        ret = np.array(features)


    return np.array(features)


def calculate_line_length(X):
    """
    Calculates the line length
    :param X: data to calculate line length for
    :type X: ndarray
    :return: line length
    :rtype: float
    """
    l = np.sum(np.abs(np.diff(X, axis=1)), axis=1)
    return l


def calculate_area(X):
    """
    Calculates area
    :param X: data to calculate area for
    :type X: ndarray 
    :return: area
    :rtype: float
    """
    return np.sum(np.abs(X), axis=1)


def calculate_energy(X):
    """
    Calculates energy
    :param X: data to calculate energy for
    :type X: ndarray
    :return: energy
    :rtype: float
    """
    return np.sum(np.square(X), axis=1)


def calculate_zero_crossings(X):
    """
    Calculates number of zero crossings
    :param X: data to calculate zero crossings for
    :type X: ndarray
    :return: number of zero crossings
    :rtype: int
    """
    X_bar = X.mean()
    zc = 0

    # TODO
    for index in range(1, len(X)):

        if X[index - 1] - X_bar > 0 and X[index] - X_bar < 0:
            zc += 1
        elif X[index - 1] - X_bar < 0 and X[index] - X_bar > 0:
            zc += 1

    return zc


def avg_frequency_mag(channel):
    """
    Calculates mean absolute value of fast fourier transform for the time window on 1 channel
    :param channel: window data from one specific channel
    :type channel: ndarray
    :return: average frequency mag
    :rtype: float
    """
    return np.mean(np.abs(fft(channel)), axis=1)


def hjorthActivity(data):
    return np.var(data, axis=1)


def hjorthMobility(data):
    gred = np.gradient(data, axis=1)
    var_gred = np.var(gred, axis=1)
    var = np.var(data, axis=1)
    q = var_gred / var
    return np.sqrt(q)


def hjorthComplexity(data):
    return hjorthMobility(np.gradient(data, axis=1)) / hjorthMobility(data)


# TODO: Normalization

# TODO: Consider using multiple different filters and getting features for each as opposed to just one
def lowpass_filtfilt(raw_eeg, fs=1000):
    """
    TODO
    :param raw_eeg:
    :type raw_eeg:
    :param fs:
    :type fs:
    :return:
    :rtype:
    """
    b, a = sig.butter(5, 300, btype='low', fs=fs)

    datas = []

    for i in range(raw_eeg.shape[1]):
        data_filtered = sig.filtfilt(b, a, raw_eeg[:, i])
        datas.append(data_filtered)
    return np.array(datas).T


def bandpass_filtfilt(raw_eeg, fs=1000):
    """
    TODO
    :param raw_eeg:
    :type raw_eeg:
    :param fs:
    :type fs:
    :return:
    :rtype:
    """

    # The sampling rate of the data glove and ecog was 1000 Hz

    fft_data = sig.fftconvolve(raw_eeg, raw_eeg[::-1], mode='same')
    # fs = len(fft_data)/1000

    filter_order = 100
    cutoff = np.arange(40, 250)

    fir_bandpass_filter = sig.firwin(numtaps=(filter_order + 1), cutoff=cutoff,
                                     pass_zero='bandpass', fs=fs)

    return sig.filtfilt(fir_bandpass_filter, 1, raw_eeg, padlen=0)


def get_number_of_windows(duration_ms, window_length_ms, stride_ms):
    """
    TODO
    :param duration_ms:
    :type duration_ms:
    :param window_length_ms:
    :type window_length_ms:
    :param stride_ms:
    :type stride_ms:
    :return:
    :rtype:
    """

    return int(((duration_ms - window_length_ms) / stride_ms) + 1)


def get_windows(data, window_length=100, stride=50, window_overlap=50, Y_bool=False, number_of_samples=None):
    """
    Divides data into windows
    :param data: data to divide into windows
    :type data: ndarray
    :param window_length: length of window
    :type window_length: int
    :param stride: distance from start of window to start of next window
    :type stride: int
    :param window_overlap: overlap between windows
    :type window_overlap: int
    :return: data divided into windows
    :rtype: ndarray
    :param Y_bool: Whether or not these windows are y values
    :type Y_bool: bool
    :param number_of_samples:
    :type number_of_samples:
    """

    if number_of_samples is None:
        number_of_samples = data.shape[0]

    number_of_channels = data.shape[1]

    number_of_windows = get_number_of_windows(number_of_samples, window_length, stride)

    window_indices = np.arange(start=(number_of_samples - window_overlap), stop=0,
                               step=-window_overlap) - 1
    window_indices = np.insert(window_indices, 0, number_of_samples - 1)
    windows_bounds = np.array([np.array([window_indices[index + 1], window_indices[index]])
                               for index in range(number_of_windows - 1)])

    if Y_bool:
        windows = np.array([data[bound[1]] for bound in windows_bounds])
    else:
        windows = np.array([[data[bound[0]:bound[1], channel] for channel in range(number_of_channels)]
                            for bound in windows_bounds])  # 590 x 61 x 50

    return windows


def get_features(filtered_window, fs=1000):
    """
    TODO
    :param filtered_window:
    :type filtered_window:
    :param fs:
    :type fs:
    :return:
    :rtype:
    """

    return np.array([feature(filtered_window) for feature in get_features.feature_funcs])


get_features.feature_funcs = [np.max, np.min, np.average, np.std,
                              calculate_line_length, calculate_area, calculate_energy, calculate_zero_crossings,
                              avg_frequency_mag]


def get_windowed_features(raw_ecog, fs, window_length, window_overlap):
    """
    Gets features for each time window
    :param raw_ecog: raw ecog data to extract features from
    :type raw_ecog: ndarray
    :param fs: frequency of data
    :type fs: int
    :param window_length: length of each window in ms
    :type window_length: int
    :param window_overlap: overlap of windows in ms
    :type window_overlap: int
    :return: features for each time window
    :rtype: ndarray
    """

    disp = window_length - window_overlap
    filtered_ecog = bandpass_filtfilt(raw_ecog)

    windows = get_windows(filtered_ecog)  # TODO: HERE
    # 590 x 61 x 50
    arr = []

    for channel_windows in windows:

        window_features = []

        for channel_window in channel_windows:
            ch_win_feats = get_features(channel_window, fs)
            window_features.append(ch_win_feats)

        arr.append(window_features)

    return np.array(arr)
