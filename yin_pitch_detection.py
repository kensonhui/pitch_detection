import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from multiprocessing import Pool

def frequencies_to_note(frequencies):
    # Reference note of A4 = 440Hz
    A4 = 440.0
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_mappings = []
    for f in frequencies:
        if f <= 0:
            note_mappings.append(None)
            continue

        n = 69 + 12 * np.log2(f / A4)
        midi_note = int(round(n))

        octaive = (midi_note // 12) - 1
        note_index = midi_note % 12
        note_name = f"{note_names[note_index]}{octaive}"
        note_mappings.append(note_name)

    return note_mappings

def note_to_frequency(note):
    # Reference note of A4 = 440Hz
    A4 = 440.0
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_index = note_names.index(note[:-1])
    octaive = int(note[-1])
    midi_note = 12 * (octaive + 1) + note_index
    return A4 * 2 ** ((midi_note - 69) / 12)

def get_note_tolerance(note_name: str, tolerance_cents=25):
    # Get the frequency range for a note with a tolerance in cents
    note_freq = note_to_frequency(note_name)
    lower_bound = note_freq * 2 ** (-tolerance_cents / 1200)
    upper_bound = note_freq * 2 ** (tolerance_cents / 1200)

    return (lower_bound, upper_bound)

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
    window_size = 256
    bounds = [20, 2000]
    
    pitches = []
    num_windows = data.shape[0] // (window_size + 3)
    print(f"{num_windows} windows")

    window_args = [(data, window_size, i * window_size, sample_rate, bounds) for i in range(num_windows)]
    
    with Pool() as p:
        pitches = p.starmap(detect_pitch, window_args)
    
    pitches = np.array(pitches)
    notes = frequencies_to_note(pitches)
    unique_notes = set(notes)
    unique_note_bins = {note: get_note_tolerance(note) for note in unique_notes}

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Plotting notes
    for note, tolerances in unique_note_bins.items():
        lower_note, upper_note = tolerances
        indices = [i for i, pitch in enumerate(pitches) if lower_note <= pitch and pitch <= upper_note]
        ax[0].scatter(np.array(indices), pitches[indices], label=note)

    ax[0].set_xlabel("Window Index")
    ax[0].set_ylabel("Pitch (Hz)")
    ax[0].set_yscale('log')
    ax[0].set_title("Detected Pitch Over Time")
    ax[0].legend()

    # Plot Audio
    ax[1].plot(np.linspace(0, len(data) / sample_rate, len(data)), data)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Amplitude")
    ax[1].set_title("Audio Signal")

    plt.show()

if __name__ == '__main__':
    main()
