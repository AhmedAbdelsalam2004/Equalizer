import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
import plotly.graph_objects as go
import uuid
import json
import os

# Load presets once at startup
@st.cache_resource
def load_presets():
    preset_path = os.path.join(os.path.dirname(__file__), "presets.json")
    try:
        with open(preset_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("presets.json not found! Place it in the same directory as app.py")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in presets.json: {e}")
        return {}

PRESETS = load_presets()

# Page configuration
st.set_page_config(
    page_title="Signal Equalizer",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    .main-header { font-size: 1.3rem; color: #1f77b4; text-align: center; margin: 0; padding: 0; }
    .band-container { background-color: #ffebee; padding: 0.3rem; border-radius: 4px; margin-bottom: 0.2rem; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .stButton button { padding: 0.2rem 0.4rem; font-size: 0.75rem; height: 1.8rem; }
    div[data-testid="stVerticalBlock"] > div { gap: 0rem; }
    div[data-testid="column"] { padding: 0.2rem; }
    .element-container { margin-bottom: 0.1rem; }
    h1, h2, h3 { margin-top: 0.2rem; margin-bottom: 0.2rem; padding: 0; }
    .stAudio { margin-top: 0.1rem; margin-bottom: 0.1rem; }
    p { margin-bottom: 0.2rem; }
    .stMarkdown { margin-bottom: 0.2rem; }
    div[data-testid="stHorizontalBlock"] { gap: 0.3rem; }
    .stSlider { margin-bottom: 0.1rem; }
    div[data-testid="stExpander"] { margin-bottom: 0.2rem; }
</style>
""", unsafe_allow_html=True)


# ===================================
# COOLEY–TUKEY FFT
# ===================================

def next_power_of_two(n):
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
# TRUE AUDIOGRAM CONVERSION
# ===================================

ISO_FREQS = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
ISO_SPL_AT_0_HL = np.array([45.0, 25.5, 11.5, 7.0, 7.0, 8.5, 12.0])


def spl_to_dB_HL(dB_SPL, freqs):
    ref_spl = np.interp(freqs, ISO_FREQS, ISO_SPL_AT_0_HL, left=ISO_SPL_AT_0_HL[0], right=ISO_SPL_AT_0_HL[-1])
    return dB_SPL - ref_spl


def magnitude_to_dB_SPL(magnitude, sample_rate):
    pressure_pa = magnitude
    pressure_pa = np.clip(pressure_pa, 1e-9, None)
    dB_SPL = 20 * np.log10(pressure_pa / 20e-6)
    return dB_SPL


# ===================================
# DYNAMIC PLOTLY PLOTS
# ===================================

def create_dynamic_fft_plot(fft_data, sample_rate, title, color):
    N = len(fft_data)
    half = N // 2
    magnitude = np.abs(fft_data[:half])
    freqs = np.linspace(0, sample_rate / 2, half)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs,
        y=magnitude,
        mode='lines',
        line=dict(color=color, width=1.5),
        name=title
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11)),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        xaxis=dict(showgrid=True, rangeslider=dict(visible=True, thickness=0.05)),
        yaxis=dict(showgrid=True),
        hovermode="x unified",
        height=250,
        margin=dict(l=35, r=20, t=30, b=30),
        font=dict(size=9)
    )
    return fig


def create_dynamic_audiogram_plot(fft_data, sample_rate, title, color):
    N = len(fft_data)
    half = N // 2
    magnitude = np.abs(fft_data[:half])
    freqs = np.linspace(0, sample_rate / 2, half)

    valid = freqs >= 100
    freqs = freqs[valid]
    magnitude = magnitude[valid]

    if len(freqs) == 0:
        freqs = np.array([100.0])
        magnitude = np.array([1e-9])

    dB_SPL = magnitude_to_dB_SPL(magnitude, sample_rate)
    dB_HL = spl_to_dB_HL(dB_SPL, freqs)
    dB_HL = np.clip(dB_HL, -10, 120)

    standard_freqs = [125, 250, 500, 1000, 2000, 4000, 8000]
    visible_freqs = [f for f in standard_freqs if f <= sample_rate / 2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs,
        y=dB_HL,
        mode='lines',
        line=dict(color=color, width=1.5),
        name=title
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11)),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Hearing Level (dB HL)",
        xaxis=dict(
            type="log",
            tickvals=visible_freqs,
            ticktext=[str(f) for f in visible_freqs],
            showgrid=True,
            range=[np.log10(100), np.log10(min(10000, sample_rate / 2))]
        ),
        yaxis=dict(
            showgrid=True,
            range=[120, -10]
        ),
        hovermode="x unified",
        height=250,
        margin=dict(l=35, r=20, t=30, b=30),
        font=dict(size=9)
    )
    return fig


def create_dynamic_spectrogram(spectrogram, sample_rate, n_fft=1024, hop_length=512, title="Spectrogram (dB SPL)"):
    """Create interactive Plotly spectrogram."""
    times = np.arange(spectrogram.shape[1]) * hop_length / sample_rate
    freqs = np.linspace(0, sample_rate / 2, spectrogram.shape[0])

    fig = go.Figure(data=go.Heatmap(
        z=spectrogram,
        x=times,
        y=freqs,
        colorscale=[
            [0.0, 'navy'],
            [0.5, 'cyan'],
            [1.0, 'yellow']
        ],
        zmin=0,
        zmax=120,
        colorbar=dict(title="dB SPL"),
        hovertemplate="Time: %{x:.2f}s<br>Frequency: %{y:.0f} Hz<br>dB SPL: %{z:.1f}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11)),
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=220,
        margin=dict(l=35, r=20, t=30, b=30),
        font=dict(size=9)
    )
    return fig

# ===================================
# CUSTOM SPECTROGRAM (STFT)
# ===================================

def custom_stft(signal, n_fft=1024, hop_length=512):
    window = np.hanning(n_fft)
    pad_len = n_fft - (len(signal) - n_fft) % hop_length
    if pad_len < n_fft:
        pad_len += n_fft
    signal_padded = np.pad(signal, (0, pad_len), mode='constant')
    frames = []
    for i in range(0, len(signal_padded) - n_fft + 1, hop_length):
        frame = signal_padded[i:i + n_fft] * window
        frames.append(frame)
    if not frames:
        frames = [np.pad(signal, (0, n_fft - len(signal)), mode='constant') * window]
    fft_frames = []
    for frame in frames:
        N = next_power_of_two(len(frame))
        frame_padded = np.pad(frame, (0, N - len(frame)), mode='constant')
        fft_result = cooley_tukey_fft(frame_padded)
        fft_frames.append(fft_result[:N // 2])
    return np.abs(np.array(fft_frames).T)


def amplitude_to_dB_SPL(S, sample_rate, n_fft=1024):
    scaling = n_fft / 2
    pressure = S / scaling
    pressure = np.clip(pressure, 1e-9, None)
    dB_SPL = 20 * np.log10(pressure / 20e-6)
    return np.clip(dB_SPL, 0, 120)


# ===================================
# Refactored Helper Function (To avoid code repetition)
# ===================================

def process_output_fft(modified_fft, original_audio_data, sample_rate):
    """Processes the modified FFT back to audio, calculates new plots, and updates state."""
    
    # Inverse FFT and Truncate
    equalized_full = cooley_tukey_ifft(modified_fft)
    equalized_audio = equalized_full[:len(original_audio_data)]
    st.session_state.equalized_audio = equalized_audio
    st.session_state.eq_applied = True

    # Re-FFT for plotting (padding if necessary)
    N_orig = len(equalized_audio)
    N_pad = next_power_of_two(N_orig)
    eq_padded = np.pad(equalized_audio, (0, N_pad - N_orig),
                       mode='constant') if N_pad > N_orig else equalized_audio
    eq_fft = cooley_tukey_fft(eq_padded)

    # Generate new plots and update session state
    st.session_state.output_fft_linear = create_dynamic_fft_plot(eq_fft, sample_rate, "Output Signal FFT", 'orange')
    st.session_state.output_audiogram = create_dynamic_audiogram_plot(eq_fft, sample_rate,
                                                             "Output Audiogram (dB HL)", 'orange')

    S_output = custom_stft(equalized_audio, n_fft=1024, hop_length=512)
    S_output_dB = amplitude_to_dB_SPL(S_output, sample_rate, n_fft=1024)
    st.session_state.output_spectrogram = create_dynamic_spectrogram(S_output_dB, sample_rate, 1024, 512,
                                                            "Output Signal (dB SPL)")

    return equalized_audio


# ===================================
# Session State
# ===================================

if 'initialized' not in st.session_state:
    st.session_state.audio_data = None
    st.session_state.sample_rate = None
    st.session_state.equalized_audio = None
    st.session_state.fft_data = None
    st.session_state.full_freqs = None
    st.session_state.eq_bands = []
    st.session_state.last_audio_hash = None
    st.session_state.input_fft_linear = None
    st.session_state.input_audiogram = None
    st.session_state.input_spectrogram = None
    st.session_state.output_fft_linear = None
    st.session_state.output_audiogram = None
    st.session_state.output_spectrogram = None
    st.session_state.eq_applied = False
    st.session_state.initialized = True


# ===================================
# Header & Upload
# ===================================

st.markdown('<h1 class="main-header">🎵 Signal Equalizer</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("#### ⚙️ Config")
    freq_scale = st.radio("Display", ["Linear FFT", "Audiogram (dB HL)"], index=0)
    mode = st.selectbox(
        "Mode:",
        ["Generic Mode", "Musical Instruments Mode", "Animal Sounds Mode", "Human Voices Mode"]
    )
    uploaded_file = st.file_uploader("Upload Audio", type=['wav', 'mp3', 'flac', 'm4a', 'aac'])


# ===================================
# Load & Preprocess (ONCE ONLY)
# ===================================

if uploaded_file is not None:
    file_hash = hash(uploaded_file.getvalue())
    if file_hash != st.session_state.last_audio_hash:
        with io.BytesIO(uploaded_file.getvalue()) as buffer:
            audio_data, sample_rate = librosa.load(buffer, sr=None)

        N_orig = len(audio_data)
        N_pad = next_power_of_two(N_orig)
        audio_padded = np.pad(audio_data, (0, N_pad - N_orig), mode='constant') if N_pad > N_orig else audio_data

        try:
            fft_data = cooley_tukey_fft(audio_padded)
            full_freqs = np.fft.fftfreq(len(fft_data), d=1 / sample_rate)

            input_fft_linear = create_dynamic_fft_plot(fft_data, sample_rate, "Input Signal FFT", 'steelblue')
            input_audiogram = create_dynamic_audiogram_plot(fft_data, sample_rate, "Input Audiogram (dB HL)", 'steelblue')

            S_input = custom_stft(audio_data, n_fft=1024, hop_length=512)
            S_input_dB = amplitude_to_dB_SPL(S_input, sample_rate, n_fft=1024)
            input_spectrogram = create_dynamic_spectrogram(S_input_dB, sample_rate, 1024, 512, "Input Signal (dB SPL)")

            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = sample_rate
            st.session_state.fft_data = fft_data
            st.session_state.full_freqs = full_freqs
            st.session_state.equalized_audio = audio_data.copy()
            st.session_state.last_audio_hash = file_hash
            st.session_state.eq_bands = []
            st.session_state.input_fft_linear = input_fft_linear
            st.session_state.input_audiogram = input_audiogram
            st.session_state.input_spectrogram = input_spectrogram
            st.session_state.output_fft_linear = None
            st.session_state.output_audiogram = None
            st.session_state.output_spectrogram = None
            st.session_state.eq_applied = False

        except Exception as e:
            st.error(f"Processing Error: {e}")


# ===================================
# Equalizer Logic
# ===================================

def apply_gain_mask_to_fft(fft_data, full_freqs, bands, sample_rate):
    N = len(fft_data)
    gain_mask = np.ones(N, dtype=np.float32)
    for band in bands:
        f0 = band["freq"]
        gain = band["gain"]
        bw = band["bandwidth"]
        low = max(0, f0 - bw / 2)
        high = min(sample_rate / 2, f0 + bw / 2)
        in_band = (np.abs(full_freqs) >= low) & (np.abs(full_freqs) <= high)
        gain_mask[in_band] *= gain
    magnitudes = np.abs(fft_data)
    phases = np.angle(fft_data)
    new_magnitudes = magnitudes * gain_mask
    return new_magnitudes * np.exp(1j * phases)


# ===================================
# Main Layout - LAYOUT SETUP
# ===================================

if st.session_state.audio_data is not None:
    col1, col2 = st.columns([1, 1])

    # Input side: FIXED and computed ONCE
    with col1:
        st.markdown("#### 📊 Input")
        st.caption(f"✅ {uploaded_file.name} | {st.session_state.sample_rate} Hz")
        st.audio(uploaded_file, format='audio/wav')
        
        if freq_scale == "Linear FFT":
            st.plotly_chart(st.session_state.input_fft_linear, use_container_width=True, config={'displayModeBar': False})
        else:
            st.plotly_chart(st.session_state.input_audiogram, use_container_width=True, config={'displayModeBar': False})


# ===================================
# Equalizer Controls (Process Logic)
# ===================================

st.markdown("#### 🎛️ Equalizer")

if st.session_state.audio_data is not None:
    if mode == "Generic Mode":

        if st.button("➕ Add Band", use_container_width=False):
            st.session_state.eq_bands.append({
                "id": str(uuid.uuid4()),
                "freq": 1000,
                "gain": 1.0,
                "bandwidth": 200
            })
            st.rerun()

        # Arrange bands in 3-column grid
        num_bands = len(st.session_state.eq_bands)
        if num_bands > 0:
            num_rows = (num_bands + 2) // 3
            for row in range(num_rows):
                cols = st.columns(3)
                for col_idx in range(3):
                    band_idx = row * 3 + col_idx
                    if band_idx < num_bands:
                        band = st.session_state.eq_bands[band_idx]
                        with cols[col_idx]:
                            st.markdown('<div class="band-container">', unsafe_allow_html=True)
                            st.markdown(f"**Band {band_idx + 1}**", unsafe_allow_html=True)
                            freq = st.slider("F", 20, st.session_state.sample_rate // 2, int(band["freq"]),
                                             key=f"freq_{band['id']}", label_visibility="collapsed")
                            st.caption(f"{freq} Hz")
                            gain = st.slider("G", 0.0, 2.0, float(band["gain"]), key=f"gain_{band['id']}", 
                                           label_visibility="collapsed")
                            st.caption(f"×{gain:.2f}")
                            bw = st.slider("BW", 10, min(5000, st.session_state.sample_rate // 2),
                                          int(band["bandwidth"]),
                                          key=f"bw_{band['id']}", label_visibility="collapsed")
                            st.caption(f"{bw} Hz")
                            band.update({"freq": freq, "gain": gain, "bandwidth": bw})
                            if st.button(f"🗑️", key=f"del_{band['id']}", use_container_width=True):
                                st.session_state.eq_bands = [b for b in st.session_state.eq_bands if b["id"] != band["id"]]
                                st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.fft_data is not None and st.session_state.eq_bands:
            modified_fft = apply_gain_mask_to_fft(
                st.session_state.fft_data,
                st.session_state.full_freqs,
                st.session_state.eq_bands,
                st.session_state.sample_rate
            )
            process_output_fft(modified_fft, st.session_state.audio_data, st.session_state.sample_rate)

    else:

        if mode not in PRESETS:
            st.warning(f"No preset for '{mode}'")
        else:
            preset_bands = []
            sources = PRESETS[mode]
            num_sources = len(sources)
            cols = st.columns(min(3, num_sources)) 
            
            for i, source_name in enumerate(sources):
                with cols[i % len(cols)]:
                    gain = st.slider(
                        source_name.capitalize(),
                        0.0, 2.0, 1.0,
                        key=f"preset_{mode}_{source_name}",
                        label_visibility="visible"
                    )
                    for center, bw in sources[source_name]:
                        preset_bands.append({
                            "freq": center,
                            "gain": gain,
                            "bandwidth": bw
                        })

            if st.session_state.fft_data is not None:
                modified_fft = apply_gain_mask_to_fft(
                    st.session_state.fft_data,
                    st.session_state.full_freqs,
                    preset_bands,
                    st.session_state.sample_rate
                )
                process_output_fft(modified_fft, st.session_state.audio_data, st.session_state.sample_rate)

else:
    st.info("Upload audio file to start")


# ===================================
# Main Layout - OUTPUT DISPLAY
# ===================================

if st.session_state.audio_data is not None:
    with col2:
        st.markdown("#### 🔊 Output")
        if st.session_state.eq_applied:
            st.caption("✅ Ready")
            out_buffer = io.BytesIO()
            sf.write(out_buffer, np.clip(st.session_state.equalized_audio, -1, 1), st.session_state.sample_rate, format='WAV')
            out_buffer.seek(0)
            st.audio(out_buffer, format='audio/wav')
            if freq_scale == "Linear FFT":
                st.plotly_chart(st.session_state.output_fft_linear, use_container_width=True, config={'displayModeBar': False})
            else:
                st.plotly_chart(st.session_state.output_audiogram, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Adjust sliders")


# ===================================
# SPECTROGRAM VIEW (DYNAMIC PLOTLY)
# ===================================

if st.session_state.audio_data is not None:
    show_spectrograms = st.checkbox("Spectrogram View", value=False)
    if show_spectrograms:
        spec_col1, spec_col2 = st.columns(2)
        with spec_col1:
            st.caption("**Input Spectrogram**")
            st.plotly_chart(st.session_state.input_spectrogram, use_container_width=True, config={'displayModeBar': False})
        with spec_col2:
            st.caption("**Output Spectrogram**")
            if st.session_state.eq_applied and st.session_state.output_spectrogram is not None:
                st.plotly_chart(st.session_state.output_spectrogram, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Adjust sliders")