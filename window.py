import pandas as pd
from numpy import array, mean, std, dstack,argmax
import itertools

def windows(series,window_size,overlap):
    start = 0
    while start+window_size <= series.shape[0]:
        yield int(start), int(start + window_size)
        start += window_size - overlap
        
        
def segment_signal(series,window_size,overlap):
    windowed_series = np.array([])
    for k in range(series.shape[1]):
        print("Windowing Column %d"%(k))
        temp_series = np.array([])
        for (start, end) in windows(series,window_size,overlap):
            if start==0:
                temp_series = series[start:end,k].T
            else:
                temp_series = np.vstack([temp_series,series[start:end,k].T])
        if k==0:
            windowed_series = temp_series
        else:
            windowed_series = np.vstack([windowed_series,temp_series])
    windowed_series = windowed_series.reshape((k+1,int(windowed_series.shape[0]/(k+1)),window_size))
    return windowed_series

def segment_Activity_Label(label_series,window_size,overlap):
    windowed_labels = np.array([])
    for (start, end) in windows(label_series,window_size,overlap):

        if start==0:
            windowed_labels = max(label_series[start:end])[0]
        else:
            windowed_labels = np.vstack([windowed_labels,max(label_series[start:end])[0]])
    return windowed_labels  