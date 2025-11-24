# app.py
import streamlit as st
import numpy as np
from scipy.io import wavfile
import io
import os

# ===================================
# COOLEY–TUKEY FFT & IFFT (needed for IFFT)
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

def cooley_tukey_ifft(X):
    N = len(X)
    X_conj = np.conj(X)
    fft_of_conj = cooley_tukey_fft(X_conj)
    return (np.conj(fft_of_conj) / N).real

# ===================================
# AUDIO HELPERS
# ===================================
def load_audio_from_buffer(buffer):
    buffer.seek(0)
    sr, data = wavfile.read(buffer)
    if data.ndim == 2:
        data = data.mean(axis=1)
    dtype = data.dtype
    if dtype.kind == 'i':
        data = data.astype(np.float64)
        info = np.iinfo(dtype)
        data = data / max(abs(info.min), info.max)
    elif dtype.kind == 'f':
        data = data.astype(np.float64)
        data = np.clip(data, -1.0, 1.0)
    else:
        raise ValueError(f"Unsupported audio data type: {dtype}")
    return sr, data

def save_audio_to_buffer(signal, sample_rate):
    signal = np.clip(signal, -1.0, 1.0)
    signal_int16 = (signal * 32767).astype(np.int16)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, signal_int16)
    buffer.seek(0)
    return buffer

# ===================================
# LOAD PRECOMPUTED FFTs (cached)
# ===================================
@st.cache_resource
def load_precomputed_sources():
    FFT_DIR = "saved_ffts"
    if not os.path.exists(FFT_DIR):
        return None
    try:
        metadata = np.load(os.path.join(FFT_DIR, "metadata.npy"), allow_pickle=True).item()
        fft_sources = []
        for i in range(1, metadata["n_sources"] + 1):
            fft = np.load(os.path.join(FFT_DIR, f"source_{i}_fft.npy"))
            fft_sources.append(fft)
        return {
            "fft_sources": fft_sources,
            "sample_rate": int(metadata["sample_rate"]),
            "max_len": int(metadata["max_len"]),
            "fft_length": int(metadata["fft_length"])
        }
    except Exception as e:
        st.error(f"Failed to load saved FFTs: {e}")
        return None

# ===================================
# STREAMLIT APP
# ===================================
st.set_page_config(page_title="Fast Source Subtractor", layout="centered")
st.title("⚡ FFT Source Subtractor (Precomputed Sources)")

sources_data = load_precomputed_sources()

if sources_data is None:
    st.error("❌ Precomputed FFTs not found! Please run `save_source_ffts.py` first.")
    st.stop()

st.success(f"✅ Loaded 4 precomputed sources | Sample rate: {sources_data['sample_rate']} Hz")

mixed_file = st.file_uploader("🔊 Upload Mixed Audio (must match sample rate)", type=["wav"])

if mixed_file:
    try:
        sr_mixed, mixed_sig = load_audio_from_buffer(mixed_file)
        if sr_mixed != sources_data["sample_rate"]:
            st.error(f"Sample rate mismatch! Expected {sources_data['sample_rate']} Hz, got {sr_mixed} Hz.")
            st.stop()

        max_len = sources_data["max_len"]
        if len(mixed_sig) < max_len:
            mixed_sig = np.pad(mixed_sig, (0, max_len - len(mixed_sig)), mode='constant')
        elif len(mixed_sig) > max_len:
            st.warning(f"Mixed audio is longer than expected ({len(mixed_sig)} > {max_len}). Truncating.")
            mixed_sig = mixed_sig[:max_len]

        # Pad to same FFT length used for sources
        target_fft_len = sources_data["fft_length"]
        if len(mixed_sig) < target_fft_len:
            mixed_padded = np.pad(mixed_sig, (0, target_fft_len - len(mixed_sig)), mode='constant')
        else:
            mixed_padded = mixed_sig[:target_fft_len]

        fft_mixed = cooley_tukey_fft(mixed_padded)

        # Checkboxes
        st.subheader("🗑️ Select Source(s) to REMOVE")
        cols = st.columns(4)
        remove_indices = []
        for i in range(4):
            with cols[i]:
                if st.checkbox(f"Remove Source {i+1}", key=f"remove_{i}"):
                    remove_indices.append(i)

        if st.button("⚡ Reconstruct (Fast IFFT)"):
            fft_residual = fft_mixed.copy()
            for idx in remove_indices:
                fft_residual -= sources_data["fft_sources"][idx]

            residual_signal = cooley_tukey_ifft(fft_residual)[:max_len]

            removed_names = [f"Source {i+1}" for i in remove_indices] or ["None"]
            st.success(f"✅ Done! Removed: {', '.join(removed_names)}")

            st.subheader("🔊 Original Mixed")
            st.audio(mixed_file)

            st.subheader(f"🔊 Result (Without {', '.join(removed_names)})")
            output_buf = save_audio_to_buffer(residual_signal, sr_mixed)
            st.audio(output_buf)

            st.download_button(
                "📥 Download Result",
                data=output_buf,
                file_name=f"output_without_{'_'.join([f's{i+1}' for i in remove_indices]) or 'none'}.wav",
                mime="audio/wav"
            )

    except Exception as e:
        st.error(f"Processing failed: {e}")
        st.exception(e)