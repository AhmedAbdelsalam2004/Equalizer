import librosa
import numpy as np
import json
import os
from scipy.signal import find_peaks

# --- CONFIGURATION ---
DEFAULT_BANDWIDTH_HZ = 1000 
TOP_N_PEAKS = 5

# Ensure these paths match where your audio files actually are
# If they are in the same folder as this script, remove "Equalizer_Data/"
ANIMAL_FILES = {
    "duck": "template_dir\duck\Free Mallard Duck Call Sound Effects (Quack) _ No Copyright(MP3_160K).mp3",
    "crow": "template_dir\crow\\11 Crow Sounds _ Calls(MP3_160K).mp3",
    "owl": "template_dir\owl\Owl Sound Effects.mp3",
    "frog": "template_dir\\frog\FROG SOUND EFFECTS - Frog Sounds.mp3",
    "turkey": "template_dir\\turkey\TURKEY SOUND EFFECTS - Turkey Gobbling(MP3_160K).mp3"
}

def analyze_audio_to_bands(audio_path, bw_size=DEFAULT_BANDWIDTH_HZ, top_n=TOP_N_PEAKS):
    """Analyzes an audio file to find dominant frequency bands."""
    if not os.path.exists(audio_path):
        # Return a placeholder so the script doesn't crash if one file is missing
        return [[500, 1000], [1000, 1000]] 

    try:
        y, sr = librosa.load(audio_path, sr=None)
        S = np.abs(librosa.stft(y))
        magnitude_spectrum = np.mean(S, axis=1)
        fft_size = S.shape[0] * 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=fft_size)
        
        # Find peaks
        peak_indices, _ = find_peaks(
            magnitude_spectrum, 
            prominence=0.05 * np.max(magnitude_spectrum), 
            distance=int(50 / (sr / fft_size))
        )
        
        # Sort by magnitude
        peak_magnitudes = magnitude_spectrum[peak_indices]
        sorted_indices = peak_indices[np.argsort(peak_magnitudes)[::-1]]
        top_indices = sorted_indices[:top_n]

        frequency_bands = []
        for index in top_indices:
            center_freq = freqs[index]
            if center_freq > 50: 
                frequency_bands.append([int(center_freq), bw_size])

        return frequency_bands
    except Exception as e:
        return []

def print_presets():
    # 1. Analyze Animals
    animal_data = {}
    print("Analyzing... (This might take a moment)\n")
    
    for animal, file_path in ANIMAL_FILES.items():
        bands = analyze_audio_to_bands(file_path)
        if bands:
            animal_data[animal] = bands
    
    # 2. Construct the JSON block for just this mode
    output_data = {
        "Animal Sounds Mode": animal_data
    }

    # 3. PRINT to terminal
    print("---------------- COPY BELOW THIS LINE ----------------")
    print(json.dumps(output_data, indent=4))
    print("---------------- COPY ABOVE THIS LINE ----------------")

if __name__ == "__main__":
    print_presets()s