import librosa
import numpy as np
import json
import os
from scipy.signal import find_peaks

# --- Configuration ---
# Set the bandwidth (BW) size for the resulting bands.
# INCREASED to 800 Hz to ensure better coverage of speech formants and harmonics.
DEFAULT_BANDWIDTH_HZ = 1000 # <-- CHANGED
# INCREASED to 5 to capture more significant frequency concentrations per source.
TOP_N_PEAKS = 8 # <-- CHANGED

def analyze_audio_to_bands(audio_path, bw_size=DEFAULT_BANDWIDTH_HZ, top_n=TOP_N_PEAKS):
    """
    Analyzes an individual audio file to find its most prominent frequency bands.
    """
    try:
        # 1. Load the audio file
        y, sr = librosa.load(audio_path, sr=None)

        # 2. Compute the magnitude spectrum (using STFT for time-averaged data)
        S = np.abs(librosa.stft(y))
        
        # 3. Average the magnitude across all time frames to get a smoothed, overall spectrum
        magnitude_spectrum = np.mean(S, axis=1)
        
        # 4. Create the frequency axis for the spectrum
        fft_size = S.shape[0] * 2  # STFT bins go up to N_FFT/2 + 1, so N_FFT is (bins-1)*2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=fft_size)
        
        # 5. Find peaks in the magnitude spectrum
        # We look for peaks with a certain prominence to filter out noise
        # A prominence threshold (e.g., 5% of max magnitude) is applied.
        peak_indices, properties = find_peaks(
            magnitude_spectrum, 
            prominence=0.05 * np.max(magnitude_spectrum), 
            distance=int(50 / (sr / fft_size)) # Ensure peaks are at least 50 Hz apart
        )
        
        # 6. Sort peaks by magnitude and select the top N
        peak_magnitudes = magnitude_spectrum[peak_indices]
        # Sort indices of peaks by magnitude in descending order
        sorted_indices = peak_indices[np.argsort(peak_magnitudes)[::-1]]
        top_indices = sorted_indices[:top_n]

        # 7. Convert indices to frequency bands
        frequency_bands = []
        for index in top_indices:
            center_freq = freqs[index]
            # Ensure the center frequency is reasonable (e.g., above 50 Hz)
            if center_freq > 50:
                 # Band is represented as [Center Frequency, Bandwidth]
                frequency_bands.append([int(center_freq), bw_size])

        # If no significant bands were found, return a fallback band
        if not frequency_bands:
            # Fallback to a mid-range speech band
            frequency_bands.append([500, bw_size]) 
            
        return frequency_bands
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return []

def generate_presets_json(voice_files_map, output_file="new_presets.json"):
    """
    Generates the entire presets JSON structure for the Human Voices Mode.
    """
    all_presets = {}
    
    # --- Human Voices Mode ---
    human_voices_data = {}
    for source_name, file_path in voice_files_map.items():
        print(f"Analyzing {source_name} from {file_path}...")
        bands = analyze_audio_to_bands(file_path)
        human_voices_data[source_name.lower()] = bands
        
    all_presets["Human Voices Mode"] = human_voices_data
    
    # --- Placeholder for other modes (You can manually add them) ---
    all_presets["Musical Instruments Mode"] = {
        "bass_guitar": [[80, 100], [400, 150]],
        "drums": [[60, 50], [2000, 500]],
        "vocals": [[300, 300], [2500, 500]],
        "piano": [[250, 200], [5000, 500]]
    }
    # NOTE: You will need to analyze audio files for these too, but we focus on voices here.
    
    print("\n--- Generated Data ---\n")
    print(json.dumps(all_presets["Human Voices Mode"], indent=4))
    print("\n----------------------\n")

    # 4. Write to JSON file
    # We load the existing presets and only update the Human Voices Mode
    try:
        if os.path.exists("Equalizer/presets.json"):
            with open("Equalizer/presets.json", 'r') as f:
                existing_presets = json.load(f)
        else:
            existing_presets = {}
    except Exception:
        existing_presets = {}

    # Merge the generated data
    existing_presets.update(all_presets)

    with open(output_file, 'w') as f:
        json.dump(existing_presets, f, indent=4)

    print(f"✅ Successfully generated and saved new data to '{output_file}'")

# --- EXECUTION ---

# Map the source names to their uploaded file paths.
# NOTE: The paths must be correct relative to where you run the script, 
# or use the absolute paths for the files you provided.
VOICE_FILES = {
    "male 1": "Equalizer_Data/2.wav",
    "male 2": "Equalizer_Data/13.wav",
    "female 1": "Equalizer_Data/female 2.wav",
    "female 2": "Equalizer_Data/female 8.wav"
}

# Run the data generation
# This line is for reference; you must run the script in your environment.
generate_presets_json(VOICE_FILES, "Equalizer/new_presets.json")