import math
import numpy as np


def NumWins(winLen, winDisp, xLen, fs):
    try:
        if winDisp != winLen:
            numwins = math.floor((xLen / fs - winLen + winDisp) / winDisp)
        else:
            numwins = math.floor((xLen / fs) / winDisp)
    except:
        print("Invalid Inputs")
    return numwins

def get_fs(times):
    times[1]
    times[0]
    return round(1/((times[1]-times[0])))



def indexHelper(x, fs, winLen, winDisp):
    # helper function to return the indices of list slices based on right-align rule for the original signal
    numWins = NumWins(winLen, winDisp, x.shape[0], fs)
    indexes = []

    for ind in range(numWins):
        start = x.shape[0] - int(ind * winDisp * fs + winLen * fs)
        end = start + int(winLen * fs)
        indexes.append([start, end])
    indexes_rev = indexes[::-1]
    return np.array(indexes_rev)


def get_windoweds(raw_ecog, fs, window_length, window_disp):
    indexes = indexHelper(raw_ecog, fs, window_length, window_disp)
    arr = np.zeros(int(len(indexes) * (fs * window_length) * raw_ecog.shape[1])).reshape(
        (len(indexes), int(fs * window_length), raw_ecog.shape[1]))
    for i, ind in enumerate(indexes):
        slice_signal = raw_ecog[ind[0]:ind[1]]
        arr[i] = slice_signal
    return arr


def get_windoweds_stack(raw_ecog, fs, window_length, window_disp):
    seg_data = []
    for j in range(len(raw_ecog)):
        indexes = indexHelper(raw_ecog[j], fs, window_length, window_disp)
        arr = []
        for i, ind in enumerate(indexes):
            slice_signal = raw_ecog[j][ind[0]:ind[1]]
            arr.append(slice_signal)
        seg_data.append(np.array(arr))
    return seg_data


def window_recover(pred, original, stride):
    out = []
    for i in range(len(pred)):
        pred_rec = np.repeat(pred[i], stride, axis=1)
        pad_len = original[i].shape[0] - pred_rec.shape[1]
        padding = np.zeros((pred_rec.shape[0], pad_len))
        pred_rec = np.hstack((padding, pred_rec))
        out.append(pred_rec.T)
    return out