import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from multiprocessing import Pool

def ACF(f, W, t, lag):
    # Vectorized ACF computation
    valid_length = min(W, len(f) - t, len(f) - (lag + t))
    return np.sum(f[t:t + valid_length] * f[lag + t:lag + t + valid_length])

def DF(f, W, t, max_lag):
    # Compute all lag values in one go
    df = np.zeros(max_lag)
    base = ACF(f, W, t, 0)
    for lag in range(max_lag):
        cross = ACF(f, W, t, lag)
        df[lag] = base + ACF(f, W, t + lag, 0) - 2 * cross
    return df

def CMNDF(df):
    # Cumulative mean normalized difference function
    cmndf = np.zeros_like(df)
    cumulative_sum = np.cumsum(df[1:])
    cmndf[1:] = df[1:] * np.arange(1, len(df)) / cumulative_sum
    cmndf[0] = 1
    return cmndf

def detect_pitch(f, W, t, sample_rate, bounds, thresh=0.1):
    max_lag = bounds[1]
    df = DF(f, W, t, max_lag)
    cmndf = CMNDF(df)
    for i in range(bounds[0], bounds[1]):
        if cmndf[i] < thresh:
            return sample_rate / i
    return sample_rate / (np.argmin(cmndf[bounds[0]:bounds[1]]) + bounds[0])

def main():
    sample_rate, f = wavfile.read('test_audio/male-C_major.wav')

    if len(f.shape) > 1:
        f = np.mean(f, axis=1) # Convert stereo to mono

    data = f.astype(np.float64)
    window_size = 1024
    bounds = [20, 2000]
    
    pitches = []
    num_windows = data.shape[0] // (window_size + 3)
    print(f"{num_windows} windows")

    window_args = [(data, window_size, i * window_size, sample_rate, bounds) for i in range(num_windows)]
    
    with Pool() as p:
        pitches = p.starmap(detect_pitch, window_args)

    plt.scatter(range(len(pitches)), pitches)
    plt.xlabel("Window Index")
    plt.ylabel("Pitch (Hz)")
    plt.title("Detected Pitch Over Time")
    plt.show()

if __name__ == '__main__':
    main()
