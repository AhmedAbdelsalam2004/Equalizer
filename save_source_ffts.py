# save_source_ffts.py
import numpy as np
from scipy.io import wavfile
import os


# ===================================
# COOLEY–TUKEY FFT IMPLEMENTATION
# ===================================
def next_power_of_two(n):
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


def bit_reverse_copy(arr):
    N = len(arr)
    if N <= 1:
        return arr.astype(complex)
    bits = N.bit_length() - 1
    rev_indices = np.zeros(N, dtype=int)
    for i in range(N):
        rev_indices[i] = int(format(i, f'0{bits}b')[::-1], 2)
    return arr[rev_indices].astype(complex)


def cooley_tukey_fft(x):
    N = len(x)
    if N == 1:
        return x.astype(complex)
    X = bit_reverse_copy(x)
    size = 2
    while size <= N:
        half = size // 2
        w_step = np.exp(-2j * np.pi / size)
        for i in range(0, N, size):
            w = 1 + 0j
            for j in range(half):
                even = X[i + j]
                odd = X[i + j + half]
                X[i + j] = even + w * odd
                X[i + j + half] = even - w * odd
                w *= w_step
        size *= 2
    return X


def pad_to_power_of_two(signal):
    N = len(signal)
    N_pad = next_power_of_two(N)
    return np.pad(signal, (0, N_pad - N), mode='constant')


# ===================================
# AUDIO LOADER (FIXED FOR FLOAT & INT)
# ===================================
def load_audio(path):
    """Load mono audio as float64 in [-1, 1]"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    sr, data = wavfile.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)  # Convert stereo to mono

    dtype = data.dtype
    if dtype.kind == 'i':  # Integer (int16, int32, etc.)
        data = data.astype(np.float64)
        info = np.iinfo(dtype)
        data = data / max(abs(info.min), info.max)
    elif dtype.kind == 'f':  # Float (float32, float64)
        data = data.astype(np.float64)
        data = np.clip(data, -1.0, 1.0)
    else:
        raise ValueError(f"Unsupported audio data type: {dtype} (kind: {dtype.kind})")

    return sr, data


# ===================================
# YOUR SOURCE FILES — CUSTOM PATHS
# ===================================
SOURCE_FILES = [
    "Equalizer_Data/rick.wav",
    "Equalizer_Data/morty.wav",
    "Equalizer_Data/tracer.wav",
    "Equalizer_Data/little girl.wav"  # Space in filename is OK in string
]

OUTPUT_DIR = "saved_ffts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================================
# MAIN: Compute and Save FFTs
# ===================================
if __name__ == "__main__":
    print("🔍 Loading source audios...")
    sources = []
    sample_rates = set()
    max_len = 0

    for f in SOURCE_FILES:
        print(f"  - Loading: {f}")
        sr, sig = load_audio(f)
        sources.append(sig)
        sample_rates.add(sr)
        max_len = max(max_len, len(sig))

    if len(sample_rates) != 1:
        raise ValueError(f"All sources must have same sample rate! Got: {sample_rates}")
    fs = sample_rates.pop()

    print(f"\n✅ Sample rate: {fs} Hz | Max length: {max_len} samples")

    # Pad all to max_len
    sources = [np.pad(s, (0, max_len - len(s)), mode='constant') for s in sources]
    # Pad to power of two for FFT
    sources_padded = [pad_to_power_of_two(s) for s in sources]

    # Compute & save FFTs
    print("\n🌀 Computing FFTs...")
    fft_sources = []
    for i, sig in enumerate(sources_padded):
        fft = cooley_tukey_fft(sig)
        fft_path = os.path.join(OUTPUT_DIR, f"source_{i + 1}_fft.npy")
        np.save(fft_path, fft)
        fft_sources.append(fft)
        print(f"  → Saved: {fft_path}")

    # Save metadata
    metadata = {
        "sample_rate": fs,
        "max_len": max_len,
        "source_files": SOURCE_FILES,
        "n_sources": 4,
        "fft_length": len(fft_sources[0])
    }
    meta_path = os.path.join(OUTPUT_DIR, "metadata.npy")
    np.save(meta_path, metadata)
    print(f"  → Saved metadata: {meta_path}")

    print("\n🎉 SUCCESS! Precomputed FFTs ready. Now run your Streamlit app.")