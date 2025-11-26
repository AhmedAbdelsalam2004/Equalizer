import librosa
import numpy as np
import json
import os
from scipy.signal import find_peaks

# --- CONFIGURATION ---
DEFAULT_BANDWIDTH_HZ = 600  # Updated to 600 based on your request
TOP_N_PEAKS = 5

# Update these to match the specific folders you listed in your desired output
ANIMAL_DIRS = {
    "duck": r"template_dir\duck",
    "cow": r"template_dir\cow",
    "tiger": r"template_dir\lion",
    "Cricket": r"template_dir\Cricket"
}

AUDIO_EXTENSIONS = ('.mp3', '.wav', '.flac', '.ogg')

def analyze_audio_to_bands(audio_path, bw_size=DEFAULT_BANDWIDTH_HZ, top_n=TOP_N_PEAKS):
    """Analyzes an audio file to find dominant frequency bands."""
    if not os.path.exists(audio_path):
        return []

    try:
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) == 0: return []
            
        S = np.abs(librosa.stft(y))
        magnitude_spectrum = np.mean(S, axis=1)
        fft_size = S.shape[0] * 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=fft_size)
        
        peak_indices, _ = find_peaks(
            magnitude_spectrum, 
            prominence=0.05 * np.max(magnitude_spectrum), 
            distance=max(1, int(50 / (sr / fft_size)))
        )
        
        if len(peak_indices) == 0: return []

        # Sort by magnitude
        peak_magnitudes = magnitude_spectrum[peak_indices]
        sorted_indices = peak_indices[np.argsort(peak_magnitudes)[::-1]]
        top_indices = sorted_indices[:top_n]

        frequency_bands = []
        for index in top_indices:
            center_freq = freqs[index]
            if center_freq > 20: 
                frequency_bands.append([int(center_freq), bw_size])

        return frequency_bands
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return []

def print_presets():
    animal_data = {}
    print("Analyzing folders... \n")
    
    for animal_category, folder_path in ANIMAL_DIRS.items():
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found for {animal_category}")
            continue

        files = os.listdir(folder_path)
        # We will collect all unique bands found in this folder
        folder_bands = []

        for file_name in files:
            if file_name.lower().endswith(AUDIO_EXTENSIONS):
                full_path = os.path.join(folder_path, file_name)
                print(f"Processing: {animal_category} -> {file_name}")
                
                # Analyze file
                new_bands = analyze_audio_to_bands(full_path)
                
                # Add to our list for this animal
                folder_bands.extend(new_bands)

        # If we found bands, add them to the dictionary using the simple key
        if folder_bands:
            # Optional: Limit to TOP_N_PEAKS total if you have many files
            # Taking the first N found across all files:
            animal_data[animal_category] = folder_bands[:TOP_N_PEAKS]

    output_data = {
        "Animal Sounds Mode": animal_data
    }

    print("\n---------------- COPY BELOW THIS LINE ----------------")
    print(json.dumps(output_data, indent=4))
    print("---------------- COPY ABOVE THIS LINE ----------------")

if __name__ == "__main__":
    print_presets()