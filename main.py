import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import sys
import os

# Step 1: Load Audio

def load_audio(file_path):
    """Loads an audio file and returns the signal and sampling rate."""
    signal, sr = librosa.load(file_path, sr=None)
    return signal, sr

# Step 2: Preprocess Audio

def preprocess_audio(signal):
    """Preprocess the audio by normalizing and removing noise."""
    signal = signal - np.mean(signal)  # Remove DC offset
    return librosa.util.normalize(signal)

# Step 3: Detect Frequencies

def detect_frequencies(signal, sr, hop_length=1024):
    """Perform Fourier Transform to detect frequencies in the audio signal in frames."""
    stft = np.abs(librosa.stft(signal, hop_length=hop_length))
    frequencies = librosa.fft_frequencies(sr=sr)
    return frequencies, stft

# Step 4: Extract Notes from Frequencies

def extract_notes_from_frequencies(frequencies, stft, threshold_factor=0.3):
    """Extract notes from frequencies using a dynamic threshold based on maximum magnitude."""
    note_frames = []
    for frame in stft.T:
        max_magnitude = np.max(frame)  # Find the maximum magnitude in the frame
        threshold = threshold_factor * max_magnitude  # Set threshold as a fraction of the max magnitude
        notes = []
        for i, magnitude in enumerate(frame):
            if magnitude > threshold:
                note = frequency_to_note(frequencies[i])
                if note:
                    notes.append(note)
        note_frames.append(set(notes))
    return note_frames


# Step 5: Map Frequencies to Notes

def frequency_to_note(frequency):
    """Converts a frequency to the closest musical note."""
    A4 = 440  # Reference frequency
    if frequency <= 0:
        return None
    semitones = 12 * np.log2(frequency / A4)
    note_number = round(semitones) + 69
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return notes[note_number % 12]

# Step 6: Identify Chords

def identify_chords(note_frames):
    """Identify the chords in each frame of notes."""
    chord_database = {
        frozenset(["C", "E", "G"]): "C Major",
        frozenset(["A", "C", "E"]): "A Minor",
        frozenset(["D", "F#", "A"]): "D Major",
        frozenset(["E", "G#", "B"]): "E Major",
        frozenset(["G", "B", "D"]): "G Major",
        frozenset(["F", "A", "C"]): "F Major",
        frozenset(["B", "D#", "F#"]): "B Major",
        frozenset(["A", "C#", "E"]): "A Major",
        frozenset(["D", "F", "A"]): "D Minor",
        frozenset(["E", "G", "B"]): "E Minor",
        frozenset(["G", "Bb", "D"]): "G Minor",
        frozenset(["F", "Ab", "C"]): "F Minor",
        frozenset(["B", "D", "F#"]): "B Minor",
        frozenset(["C", "Eb", "G"]): "C Minor",
        frozenset(["C", "E", "G", "B"]): "C Major 7th",
        frozenset(["A", "C", "E", "G"]): "A Minor 7th",
        frozenset(["D", "F#", "A", "C"]): "D Major 7th",
        frozenset(["E", "G#", "B", "D"]): "E Major 7th",
        frozenset(["G", "B", "D", "F"]): "G Major 7th",
        frozenset(["F", "A", "C", "E"]): "F Major 7th",
        frozenset(["C#", "F", "G#"]): "C# Major",
        frozenset(["F#", "A#", "C#"]): "F# Major",
        frozenset(["G#", "C", "D#"]): "G# Major",
        frozenset(["A#", "D", "F"]): "A# Major",
        # Add more chords as needed
    }
    chords = []
    for notes in note_frames:
        chord = chord_database.get(frozenset(notes), "Unknown Chord")
        chords.append(chord)
    return chords

# Step 7: Map to Fretboard

def map_to_fretboard(notes):
    """Map notes to guitar fretboard positions."""
    fretboard = {
        "E": ["F", "F#", "G", "G#", "A", "A#", "B", "C", "C#", "D", "D#", "E"],
        "A": ["A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A"],
        "D": ["D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C", "C#", "D"],
        "G": ["G#", "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G"],
        "B": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        "e": ["F", "F#", "G", "G#", "A", "A#", "B", "C", "C#", "D", "D#", "E"],
    }
    positions = {}
    for string, frets in fretboard.items():
        positions[string] = [i for i, note in enumerate(frets) if note in notes]
    return positions

# Step 8: Visualize Chord Shapes

def visualize_chords(note_frames):
    """Visualize the chord progressions on a guitar fretboard."""
    for i, notes in enumerate(note_frames):
        fret_positions = map_to_fretboard(notes)
        print(f"Frame {i + 1}: Notes {notes}")
        for string, frets in fret_positions.items():
            print(f"  String {string}: Frets {frets}")

def split_audio_by_frames(signal, sr, frame_length, hop_length, frames_per_segment=50):
    """Split the audio into segments based on the number of frames."""
    # Calculate the time duration per frame
    frame_duration = hop_length / sr  # Time in seconds per frame
    segment_duration = frames_per_segment * frame_duration  # Total time per segment
    
    # Calculate samples per segment
    samples_per_segment = int(segment_duration * sr)
    
    # Split the signal
    segments = [
        signal[i:i + samples_per_segment] 
        for i in range(0, len(signal), samples_per_segment)
    ]
    return segments


def save_audio_segment(segment, sr, segment_index):
    """Save an audio segment as a WAV file."""
    file_name = f"output/segment_{segment_index}.wav"
    sf.write(file_name, segment, sr)
    print(f"Saved: {file_name}")

def plot_frames_frequencies(frequencies, stft, segment_idx, output_dir="output"):
    """Plot the frequencies for the first frame of a segment and save as an image file."""
    for i in range(len(stft[0])):
        first_frame = stft[:, i]
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, first_frame)
        plt.title(f"Frequency Spectrum - First Frame of Segment {segment_idx + 1}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.xlim(0, max(frequencies))  # Focus on the visible frequency range
        plt.grid(True)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plot as an image file (PNG)
        plot_filename = f"{output_dir}/frequency_spectrum_segment_{segment_idx + 1}_{i}.png"
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to free up memory
        print(f"Saved plot: {plot_filename}")

# Function to extract and listen to a frame from the original audio signal
def listen_to_frame(signal, sr, frame_index, frame_length, hop_length):
    """
    Extracts a specific frame from the original signal and plays it.
    
    Args:
        signal (np.array): The audio signal.
        sr (int): The sampling rate of the audio.
        frame_index (int): The index of the frame to extract.
        frame_length (int): The length of each frame in samples.
        hop_length (int): The number of samples between consecutive frames.
    
    Returns:
        None
    """
    start_sample = frame_index * hop_length
    end_sample = start_sample + frame_length
    
    # Extract the specific frame from the audio signal
    frame_signal = signal[start_sample:end_sample]
    
    # Save the frame as a .wav file
    sf.write(f"frame_{frame_index}.wav", frame_signal, sr)
    
    # Play the extracted frame directly (if in a Jupyter notebook)
    return ipd.Audio(frame_signal, rate=sr)


# Main Function
if __name__ == "__main__":
    file_path = "test_audio/neon_intro.m4a"  # Replace with your file path

    output_file = "output/chords.txt"
    # Load and preprocess audio
    signal, sr = load_audio(file_path)
    signal = preprocess_audio(signal)

    # Detect frequencies and identify notes
    hop_length = 2048
    frequencies, stft = detect_frequencies(signal, sr, hop_length=hop_length)
    note_frames = extract_notes_from_frequencies(frequencies, stft)

    # Split the audio into segments
    segments = split_audio_by_frames(signal, sr, stft.shape[1], hop_length)

    # Process each segment
    with open(output_file, "w+") as f:
        sys.stdout = f  # Redirect stdout to the file
        for idx, segment in enumerate(segments):
            print(f"\nSegment {idx + 1}:")
            segment_frequencies, segment_stft = detect_frequencies(segment, sr, hop_length=hop_length)
            segment_note_frames = extract_notes_from_frequencies(segment_frequencies, segment_stft, threshold_factor=0.2)
            segment_chords = identify_chords(segment_note_frames)

            for i, (notes, chord) in enumerate(zip(segment_note_frames, segment_chords)):
                print(f"  Frame {i + 1}: Notes {', '.join(sorted(notes))} -> Chord: {chord}")
            
            if (idx + 1) % 5 == 0:
                save_audio_segment(segment, sr, idx + 1)
                plot_frames_frequencies(segment_frequencies, segment_stft, idx)
    # Visualize chord progressions
    #visualize_chords(note_frames)