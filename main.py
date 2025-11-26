import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
import plotly.graph_objects as go
import uuid
import json
import os
import base64
import streamlit.components.v1 as components
from audio_router import AIAudioProcessor

st.set_page_config(
    page_title="Signal Equalizer",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
<style>
    .main > div { padding-top: 0.5rem; }
    .block-container { padding-top: 1rem; padding-bottom: 0rem; max-width: 100%; }
    .main-header { font-size: 1.5rem; color: #1f77b4; text-align: center; margin-bottom: 0.5rem; margin-top: 0; }
    h1, h2, h3, h4 { margin-top: 0.3rem !important; margin-bottom: 0.3rem !important; }
    .stSlider { padding-top: 0rem !important; padding-bottom: 0rem !important; }
    .stSlider label { font-size: 13px !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.4rem !important; }
    div[data-testid="column"] { padding: 0rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_presets():
    preset_path = os.path.join(os.path.dirname(__file__), "presets.json")
    try:
        with open(preset_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def load_generic_config():
    """Loads saved generic mode slider configuration."""
    config_path = os.path.join(os.path.dirname(__file__), "generic_config.json")
    try:
        with open(config_path, "r") as f:
            bands = json.load(f)
            for band in bands:
                band["id"] = str(uuid.uuid4())
            return bands
    except (FileNotFoundError, json.JSONDecodeError):
        return [{
            "id": str(uuid.uuid4()),
            "freq": 1000,
            "gain": 1.0,
            "bandwidth": 200
        }]

def save_generic_config(bands):
    """Saves current generic mode slider configuration to JSON."""
    config_path = os.path.join(os.path.dirname(__file__), "generic_config.json")
    clean_bands = [{k: v for k, v in b.items() if k != 'id'} for b in bands]
    try:
        with open(config_path, "w") as f:
            json.dump(clean_bands, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

@st.cache_resource
def load_precomputed_ffts():
    fft_dir = os.path.join(os.path.dirname(__file__), "saved_ffts")
    if not os.path.exists(fft_dir):
        return None
    try:
        metadata_path = os.path.join(fft_dir, "metadata.npy")
        if not os.path.exists(metadata_path):
            return None
        metadata = np.load(metadata_path, allow_pickle=True).item()
        fft_sources = []
        for i in range(1, metadata["n_sources"] + 1):
            fft = np.load(os.path.join(fft_dir, f"source_{i}_fft.npy"))
            fft_sources.append(fft)
        return {
            "fft_sources": fft_sources,
            "sample_rate": int(metadata["sample_rate"]),
            "max_len": int(metadata["max_len"]),
            "fft_length": int(metadata["fft_length"])
        }
    except Exception as e:
        print(f"Error loading FFTs: {e}")
        return None

PRESETS = load_presets()
PRECOMPUTED_DATA = load_precomputed_ffts()


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


ISO_FREQS = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
ISO_SPL_AT_0_HL = np.array([45.0, 25.5, 11.5, 7.0, 7.0, 8.5, 12.0])

def spl_to_dB_HL(dB_SPL, freqs):
    ref_spl = np.interp(freqs, ISO_FREQS, ISO_SPL_AT_0_HL, left=ISO_SPL_AT_0_HL[0], right=ISO_SPL_AT_0_HL[-1])
    return dB_SPL - ref_spl

def magnitude_to_dB_SPL(magnitude, sample_rate):
    pressure_pa = np.clip(magnitude, 1e-9, None)
    dB_SPL = 20 * np.log10(pressure_pa / 20e-6)
    return dB_SPL


@st.cache_data
def create_fft_fig(fft_data_hash, fft_data_bytes, sample_rate, title, color):
    fft_data = np.frombuffer(fft_data_bytes, dtype=np.complex128)
    N = len(fft_data)
    half = N // 2
    magnitude = np.abs(fft_data[:half])
    freqs = np.linspace(0, sample_rate / 2, half)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=magnitude, mode='lines',
        line=dict(color=color, width=1.5), name=title,
        fill='tozeroy'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
        margin=dict(l=35, r=10, t=30, b=10),
        xaxis=dict(showgrid=True, gridcolor='#eee'),
        yaxis=dict(showgrid=True, gridcolor='#eee'),
        plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", height=200
    )
    return fig

@st.cache_data
def create_audiogram_fig(fft_data_hash, fft_data_bytes, sample_rate, title, color):
    fft_data = np.frombuffer(fft_data_bytes, dtype=np.complex128)
    N = len(fft_data)
    half = N // 2
    magnitude = np.abs(fft_data[:half])
    freqs = np.linspace(0, sample_rate / 2, half)
    
    # Filter for standard audiometric frequencies
    standard_freqs = np.array([250, 500, 1000, 2000, 4000, 8000])
    valid_indices = []
    for sf in standard_freqs:
        if sf < sample_rate / 2:
            # Find closest frequency bin
            idx = (np.abs(freqs - sf)).argmin()
            valid_indices.append(idx)
            
    if not valid_indices:
        plot_freqs = np.array([1000.0])
        plot_magnitude = np.array([1e-9])
    else:
        plot_freqs = freqs[valid_indices]
        plot_magnitude = magnitude[valid_indices]

    dB_SPL = magnitude_to_dB_SPL(plot_magnitude, sample_rate)
    dB_HL = spl_to_dB_HL(dB_SPL, plot_freqs)
    # Clip to standard audiogram range
    dB_HL = np.clip(dB_HL, -10, 120)

    fig = go.Figure()
    
    # Plot with markers and lines, mimicking the style (using 'O' for this data)
    fig.add_trace(go.Scatter(
        x=plot_freqs, y=dB_HL, 
        mode='lines+markers',
        line=dict(color=color, width=2),
        marker=dict(symbol='circle-open', size=10, line=dict(width=2)),
        name=title
    ))

    # Define hearing loss severity ranges for Y-axis labels
    y_tick_vals = [0, 20, 40, 55, 70, 90, 120]
    y_tick_text = ["Normal", "Mild", "Moderate", "Moderately Severe", "Severe", "Profound", ""]
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#f3f4f6')),
        xaxis=dict(
            type="log",
            tickvals=standard_freqs,
            ticktext=[str(int(f)) for f in standard_freqs],
            title=dict(text="Frequency (Hz)", font=dict(color='#9ca3af')),
            tickfont=dict(color='#9ca3af'),
            showgrid=True, gridcolor='#374151',
            range=[np.log10(125), np.log10(10000)] # Slightly wider range for aesthetics
        ),
        yaxis=dict(
            title=dict(text="Hearing Level (dB HL)", font=dict(color='#9ca3af')),
            tickfont=dict(color='#9ca3af'),
            range=[125, -15], # Reversed range from -15 to 125
            showgrid=True, gridcolor='#374151',
            zerolinecolor='#374151',
            tickvals=np.arange(-10, 130, 10)
        ),
        # Add a secondary Y-axis for the severity labels on the right
        yaxis2=dict(
            overlaying='y',
            side='right',
            tickvals=y_tick_vals,
            ticktext=y_tick_text,
            tickfont=dict(color='#9ca3af', size=11),
            range=[125, -15], # Match primary Y-axis range
            showgrid=False,
            zeroline=False
        ),
        margin=dict(l=50, r=80, t=40, b=30), # Increase right margin for labels
        plot_bgcolor='rgba(0,0,0,0)', # Transparent background
        paper_bgcolor='rgba(0,0,0,0)', # Transparent paper
        hovermode="closest",
        height=300,
        legend=dict(font=dict(color='#f3f4f6'))
    )
    return fig

@st.cache_data
def create_spectrogram_fig(spec_data_hash, spec_data_bytes, spec_shape, sample_rate, hop_length, title):
    spectrogram_data = np.frombuffer(spec_data_bytes, dtype=np.float64).reshape(spec_shape)
    times = np.arange(spectrogram_data.shape[1]) * hop_length / sample_rate
    freqs = np.linspace(0, sample_rate / 2, spectrogram_data.shape[0])
    fig = go.Figure(data=go.Heatmap(
        z=spectrogram_data, x=times, y=freqs,
        colorscale='Viridis', zmin=0, zmax=120,
        colorbar=dict(title="dB SPL"),
        hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.0f} Hz<br>dB: %{z:.1f}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
        margin=dict(l=35, r=10, t=30, b=10), height=180
    )
    return fig


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


def apply_gain_mask_to_fft(fft_data, full_freqs, bands, sample_rate, use_multiplication=False):
    """
    Applies frequency band gains to the FFT data.
    
    Args:
        use_multiplication (bool): 
            If False (Generic Mode): Later bands OVERWRITE earlier ones (=).
            If True (Preset Mode): Bands MULTIPLY together (*=) to allow stacking mutes.
    """
    N = len(fft_data)
    gain_mask = np.ones(N, dtype=np.float32)
    
    for band in bands:
        f0 = band["freq"]
        gain = band["gain"]
        bw = band["bandwidth"]
        low = max(0, f0 - bw / 2)
        high = min(sample_rate / 2, f0 + bw / 2)
        in_band = (np.abs(full_freqs) >= low) & (np.abs(full_freqs) <= high)
        
        # --- FIXED: Logic Branching based on Mode ---
        if use_multiplication:
            gain_mask[in_band] *= gain # Accumulate gains (Preset Mode)
        else:
            gain_mask[in_band] = gain  # Overwrite gains (Generic Mode)

    magnitudes = np.abs(fft_data)
    phases = np.angle(fft_data)
    new_magnitudes = magnitudes * gain_mask
    return new_magnitudes * np.exp(1j * phases)

def apply_linear_subtraction(fft_mix, gains, precomputed_data):
    fft_out = fft_mix.copy()
    sources = precomputed_data["fft_sources"]
    n_bins = len(fft_out)
    for i, gain in enumerate(gains):
        if i < len(sources):
            src_fft = sources[i]
            limit = min(n_bins, len(src_fft))
            factor = gain - 1.0
            if abs(factor) > 1e-5:
                fft_out[:limit] += factor * src_fft[:limit]
    return fft_out


@st.cache_data
def create_audio_player_html(audio_b64_hash, audio_b64, player_id):
    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
/* Base */
body {{ margin: 0; padding: 0; font-family: 'Inter', sans-serif; background: transparent; overflow: hidden; }}
* {{ box-sizing: border-box; }}

/* Compact Card */
.audio-player {{
  background: #111827;
  border: 1px solid #374151;
  border-radius: 12px;
  padding: 12px;
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  color: #f3f4f6;
  user-select: none; /* Prevent text selection while dragging */
}}

/* Visualization */
.canvas-wrap {{ 
  position: relative; 
  background: #1f2937;
  border-radius: 8px;
  overflow: hidden;
  flex-grow: 1;
  min-height: 50px;
  margin-bottom: 4px;
  box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.2);
}}

canvas {{ display: block; width: 100%; height: 100%; }}

/* Status Badge */
.status {{ 
  position: absolute; top: 6px; right: 6px; 
  font-size: 9px; font-weight: 700; text-transform: uppercase;
  color: #60a5fa; background: rgba(17, 24, 39, 0.8); 
  padding: 2px 6px; border-radius: 99px; 
  border: 1px solid rgba(96, 165, 250, 0.2);
}}

/* Timeline / Seek Bar */
.progress-container {{
  width: 100%;
  height: 24px;
  display: flex;
  align-items: center;
  cursor: pointer;
  position: relative;
  margin-bottom: 4px;
  touch-action: none; /* Prevent scrolling on mobile while seeking */
}}

.progress-track {{
  width: 100%;
  height: 4px;
  background: #374151;
  border-radius: 2px;
  position: relative;
}}

.progress-fill {{
  height: 100%;
  background: #ef4444; 
  width: 0%;
  border-radius: 2px;
  pointer-events: none;
  /* Removed transition here to make dragging instant/snappy */
}}

.progress-cursor {{
  width: 12px;
  height: 12px;
  background: #ef4444; 
  border: 2px solid white;
  border-radius: 50%;
  position: absolute;
  top: 50%;
  left: 0%;
  transform: translate(-50%, -50%);
  box-shadow: 0 2px 4px rgba(0,0,0,0.5);
  pointer-events: none;
  /* Only animate transform (scale), not left position, for instant drag response */
  transition: transform 0.1s; 
}}

.progress-container:hover .progress-cursor {{
  transform: translate(-50%, -50%) scale(1.2);
}}

/* Controls */
.controls {{ display: flex; align-items: center; justify-content: space-between; height: 36px; }}
.time-info {{ font-variant-numeric: tabular-nums; font-size: 11px; color: #9ca3af; font-weight: 500; }}

/* Buttons */
.btn {{ display: flex; align-items: center; justify-content: center; border: none; cursor: pointer; border-radius: 50%; color: white; transition: 0.2s; }}
.btn-main {{ width: 36px; height: 36px; background: linear-gradient(135deg, #3b82f6, #2563eb); box-shadow: 0 4px 6px rgba(37,99,235,0.3); }}
.btn-main:hover {{ transform: scale(1.05); }}
.btn-sec {{ width: 28px; height: 28px; background: #374151; color: #e5e7eb; }}
.btn-sec:hover {{ background: #4b5563; color: white; }}
.btn svg {{ fill: currentColor; }}

</style>
</head>
<body>

<div class="audio-player">
  <div class="canvas-wrap">
    <canvas id="canvas-{player_id}"></canvas>
    <div class="status" id="status-{player_id}">Loading...</div>
  </div>
  
  <div class="progress-container" id="progress-{player_id}">
    <div class="progress-track">
      <div class="progress-fill" id="fill-{player_id}"></div>
      <div class="progress-cursor" id="cursor-{player_id}"></div>
    </div>
  </div>
  
  <div class="controls">
    <button class="btn btn-sec" id="stop-btn-{player_id}" title="Stop">
      <svg width="12" height="12" viewBox="0 0 24 24"><path d="M6 6h12v12H6z"/></svg>
    </button>

    <button class="btn btn-main" id="play-btn-{player_id}" title="Play/Pause">
      <svg id="icon-play-{player_id}" width="16" height="16" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
      <svg id="icon-pause-{player_id}" width="16" height="16" viewBox="0 0 24 24" style="display:none"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
    </button>

    <div class="time-info" id="time-{player_id}">0:00 / 0:00</div>
  </div>
</div>

<script>
(function() {{
  const canvas = document.getElementById('canvas-{player_id}');
  const ctx = canvas.getContext('2d');
  const playBtn = document.getElementById('play-btn-{player_id}');
  const stopBtn = document.getElementById('stop-btn-{player_id}');
  const iconPlay = document.getElementById('icon-play-{player_id}');
  const iconPause = document.getElementById('icon-pause-{player_id}');
  const timeDisplay = document.getElementById('time-{player_id}');
  const statusDisplay = document.getElementById('status-{player_id}');
  
  const progressBar = document.getElementById('progress-{player_id}');
  const progressFill = document.getElementById('fill-{player_id}');
  const progressCursor = document.getElementById('cursor-{player_id}');
  
  let audioContext, analyser, audioBuffer, source;
  let isPlaying = false, animationId = null, startTime = 0, pauseTime = 0;
  let isDragging = false; // NEW: Track drag state
  const audioData = 'data:audio/wav;base64,{audio_b64}';
  
  function setPlayState(playing) {{
    iconPlay.style.display = playing ? 'none' : 'block';
    iconPause.style.display = playing ? 'block' : 'none';
  }}

  function formatTime(s) {{ 
    const m = Math.floor(s/60); 
    const sec = Math.floor(s%60); 
    return m + ':' + (sec < 10 ? '0' : '') + sec; 
  }}

  function getCurrentTime() {{
    if (!audioBuffer) return 0;
    const dur = audioBuffer.duration;
    if (isPlaying) {{
        return Math.min(audioContext.currentTime - startTime + pauseTime, dur);
    }}
    return Math.min(pauseTime, dur);
  }}

  function updateUI() {{
    // Don't fight the mouse while dragging
    if (isDragging || !audioBuffer) return; 

    const dur = audioBuffer.duration;
    const cur = getCurrentTime();
    
    // Update Text
    timeDisplay.textContent = formatTime(cur) + ' / ' + formatTime(dur);
    
    // Update Seek Bar
    const pct = (cur / dur) * 100;
    progressFill.style.width = pct + '%';
    progressCursor.style.left = pct + '%';
  }}

  function resizeCanvas() {{ 
    const r = canvas.getBoundingClientRect(); 
    canvas.width = r.width * 2; canvas.height = r.height * 2;
    ctx.scale(2, 2); 
  }}

  function drawVisualization() {{
    if (!analyser) return;
    const bufLen = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufLen);
    analyser.getByteFrequencyData(dataArray);
    const w = canvas.getBoundingClientRect().width;
    const h = canvas.getBoundingClientRect().height;
    ctx.clearRect(0, 0, w, h);
    
    // Gradient
    const gradient = ctx.createLinearGradient(0, h, 0, 0);
    gradient.addColorStop(0, '#8b5cf6'); gradient.addColorStop(1, '#06b6d4');
    
    const barWidth = (w / bufLen) * 2.5;
    let x = 0;
    ctx.fillStyle = gradient;
    
    for (let i = 0; i < bufLen; i++) {{
      const barHeight = (dataArray[i] / 255) * h;
      ctx.beginPath();
      ctx.roundRect(x, h - barHeight, barWidth, barHeight, [3,3,0,0]);
      ctx.fill();
      x += barWidth + 1;
    }}
    
    updateUI();
    
    if (isPlaying) {{
        if (getCurrentTime() >= audioBuffer.duration) {{
            stop();
        }} else {{
            animationId = requestAnimationFrame(drawVisualization);
        }}
    }}
  }}
  
  async function initAudio() {{
    if (audioContext) return;
    try {{
      const response = await fetch(audioData);
      const arrayBuffer = await response.arrayBuffer();
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioContext = new AudioContext();
      audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 128;
      analyser.smoothingTimeConstant = 0.85;
      analyser.connect(audioContext.destination);
      statusDisplay.textContent = 'Ready';
      statusDisplay.style.color = '#34d399';
      updateUI();
    }} catch (error) {{
      console.error(error);
      statusDisplay.textContent = 'Error';
      statusDisplay.style.color = '#f87171';
    }}
  }}
  
  async function play() {{
    if (!audioContext) await initAudio();
    if (isPlaying) return;
    if (audioContext.state === 'suspended') await audioContext.resume();

    if (pauseTime >= audioBuffer.duration) {{
        pauseTime = 0;
    }}

    source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(analyser);
    
    startTime = audioContext.currentTime;
    source.start(0, pauseTime);
    
    isPlaying = true;
    setPlayState(true);
    statusDisplay.textContent = 'Playing';
    drawVisualization();
  }}
  
  function pause() {{
    if (!isPlaying || !source) return;
    source.stop();
    pauseTime += audioContext.currentTime - startTime;
    isPlaying = false;
    setPlayState(false);
    statusDisplay.textContent = 'Paused';
    if (animationId) cancelAnimationFrame(animationId);
    updateUI();
  }}
  
  function stop() {{
    if (source) {{
      try {{ source.stop(); }} catch(e) {{}}
    }}
    isPlaying = false; 
    pauseTime = 0;
    setPlayState(false);
    statusDisplay.textContent = 'Stopped';
    if (animationId) cancelAnimationFrame(animationId);
    ctx.clearRect(0, 0, canvas.getBoundingClientRect().width, canvas.getBoundingClientRect().height);
    updateUI();
  }}

  // --- DRAG / SLIDER LOGIC ---
  
  function getProgressFromEvent(e) {{
    const rect = progressBar.getBoundingClientRect();
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const x = clientX - rect.left;
    return Math.max(0, Math.min(1, x / rect.width));
  }}

  function updateVisualsForDrag(pct) {{
     progressFill.style.width = (pct * 100) + '%';
     progressCursor.style.left = (pct * 100) + '%';
     if (audioBuffer) {{
       const t = pct * audioBuffer.duration;
       timeDisplay.textContent = formatTime(t) + ' / ' + formatTime(audioBuffer.duration);
     }}
  }}

  async function handleSeek(pct) {{
     if (!audioBuffer) return;
     const targetTime = pct * audioBuffer.duration;
     
     // Initialize if needed (clicking bar before playing)
     if (!audioContext) await initAudio();
     
     if (isPlaying) {{
        source.stop();
        source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(analyser);
        pauseTime = targetTime;
        startTime = audioContext.currentTime;
        source.start(0, pauseTime);
     }} else {{
        pauseTime = targetTime;
        updateUI();
     }}
  }}

  function onMouseDown(e) {{
    isDragging = true;
    const pct = getProgressFromEvent(e);
    updateVisualsForDrag(pct);
  }}

  function onMouseMove(e) {{
    if (!isDragging) return;
    const pct = getProgressFromEvent(e);
    updateVisualsForDrag(pct);
  }}

  async function onMouseUp(e) {{
    if (!isDragging) return;
    isDragging = false;
    const pct = getProgressFromEvent(e);
    await handleSeek(pct);
  }}

  // Add Listeners
  progressBar.addEventListener('mousedown', onMouseDown);
  document.addEventListener('mousemove', onMouseMove);
  document.addEventListener('mouseup', onMouseUp);
  
  // Touch support for mobile
  progressBar.addEventListener('touchstart', onMouseDown);
  document.addEventListener('touchmove', onMouseMove);
  document.addEventListener('touchend', onMouseUp);
  
  playBtn.onclick = async () => {{ if (isPlaying) pause(); else await play(); }};
  stopBtn.onclick = stop;
  
  window.addEventListener('load', () => {{ resizeCanvas(); initAudio(); }});
  window.addEventListener('resize', resizeCanvas);
}})();
</script>
</body>
</html>
"""

def init_session_state():
    defaults = {
        'audio_data': None,
        'sample_rate': None,
        'fft_data': None,
        'full_freqs': None,
        'last_audio_hash': None,
        'input_audio_b64': None,
        'input_fft_hash': None,
        'input_spec_data': None,
        'eq_bands': load_generic_config(), # Load saved config on startup
        'output_audio_b64': None,
        'output_fft_data': None,
        'output_fft_hash': None,
        'output_spec_data': None,
        'ai_audio_b64': None,
        'ai_fft_data': None,
        'ai_fft_hash': None,
        'ai_spec_data': None,
        'ai_server_status': None,
        'current_mode': 'Generic Mode',
        'freq_scale': 'Linear FFT',
        'generic_hash': None,
        'preset_hashes': {},
        'pending_ai_request': None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_session_state()


def compute_audio_data(audio_data, sample_rate):
    N_orig = len(audio_data)
    N_pad = next_power_of_two(N_orig)
    audio_padded = np.pad(audio_data, (0, N_pad - N_orig), mode='constant') if N_pad > N_orig else audio_data
    fft_data = cooley_tukey_fft(audio_padded)
    full_freqs = np.fft.fftfreq(len(fft_data), d=1 / sample_rate)
    
    input_bytes = io.BytesIO()
    sf.write(input_bytes, audio_data, sample_rate, format='WAV')
    input_bytes.seek(0)
    input_audio_b64 = base64.b64encode(input_bytes.read()).decode()
    
    S_input = custom_stft(audio_data, n_fft=1024, hop_length=512)
    S_input_dB = amplitude_to_dB_SPL(S_input, sample_rate, n_fft=1024)
    
    return fft_data, full_freqs, input_audio_b64, S_input_dB


def process_new_audio(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = hash(file_bytes)
    
    if file_hash == st.session_state.last_audio_hash:
        return False
    
    with io.BytesIO(file_bytes) as buffer:
        audio_data, sample_rate = librosa.load(buffer, sr=None)
    
    fft_data, full_freqs, input_audio_b64, S_input_dB = compute_audio_data(audio_data, sample_rate)
    fft_hash = hash(fft_data.tobytes())
    
    st.session_state.audio_data = audio_data
    st.session_state.sample_rate = sample_rate
    st.session_state.fft_data = fft_data
    st.session_state.full_freqs = full_freqs
    st.session_state.last_audio_hash = file_hash
    st.session_state.input_audio_b64 = input_audio_b64
    st.session_state.input_fft_hash = fft_hash
    st.session_state.input_spec_data = S_input_dB
    
    # Initialize Output to match Input immediately
    st.session_state.output_audio_b64 = input_audio_b64
    st.session_state.output_fft_data = fft_data
    st.session_state.output_fft_hash = fft_hash
    st.session_state.output_spec_data = S_input_dB
    
    # Reset processing states
    st.session_state.ai_audio_b64 = None
    st.session_state.ai_fft_data = None
    st.session_state.ai_fft_hash = None
    st.session_state.ai_spec_data = None
    st.session_state.generic_hash = None # Reset hash to force recalculation if generic mode active
    st.session_state.preset_hashes = {}
    st.session_state.pending_ai_request = None
    
    # --- CHANGED: Apply Generic EQ immediately if saved settings exist ---
    if st.session_state.current_mode == 'Generic Mode' and st.session_state.eq_bands:
        process_generic_eq(st.session_state.eq_bands)
    
    return True


def update_output(equalized_audio):
    sample_rate = st.session_state.sample_rate
    
    N_orig = len(equalized_audio)
    N_pad = next_power_of_two(N_orig)
    eq_padded = np.pad(equalized_audio, (0, N_pad - N_orig), mode='constant') if N_pad > N_orig else equalized_audio
    eq_fft = cooley_tukey_fft(eq_padded)
    
    output_bytes = io.BytesIO()
    sf.write(output_bytes, equalized_audio, sample_rate, format='WAV')
    output_bytes.seek(0)
    output_audio_b64 = base64.b64encode(output_bytes.read()).decode()
    
    S_output = custom_stft(equalized_audio, n_fft=1024, hop_length=512)
    S_output_dB = amplitude_to_dB_SPL(S_output, sample_rate, n_fft=1024)
    
    st.session_state.output_audio_b64 = output_audio_b64
    st.session_state.output_fft_data = eq_fft
    st.session_state.output_fft_hash = hash(eq_fft.tobytes())
    st.session_state.output_spec_data = S_output_dB


def update_ai_output(ai_audio):
    sample_rate = st.session_state.sample_rate
    original_len = len(st.session_state.audio_data)
    
    if len(ai_audio) < original_len:
        ai_audio = np.pad(ai_audio, (0, original_len - len(ai_audio)), mode='constant')
    elif len(ai_audio) > original_len:
        ai_audio = ai_audio[:original_len]
    
    N_ai = len(ai_audio)
    N_pad = next_power_of_two(N_ai)
    ai_padded = np.pad(ai_audio, (0, N_pad - N_ai), mode='constant') if N_pad > N_ai else ai_audio
    ai_fft = cooley_tukey_fft(ai_padded)
    
    ai_bytes = io.BytesIO()
    sf.write(ai_bytes, ai_audio, sample_rate, format='WAV')
    ai_bytes.seek(0)
    ai_audio_b64 = base64.b64encode(ai_bytes.read()).decode()
    
    S_ai = custom_stft(ai_audio, n_fft=1024, hop_length=512)
    S_ai_dB = amplitude_to_dB_SPL(S_ai, sample_rate, n_fft=1024)
    
    st.session_state.ai_audio_b64 = ai_audio_b64
    st.session_state.ai_fft_data = ai_fft
    st.session_state.ai_fft_hash = hash(ai_fft.tobytes())
    st.session_state.ai_spec_data = S_ai_dB


def process_generic_eq(bands):
    if not bands or st.session_state.fft_data is None:
        return False
    
    bands_tuple = tuple((b["freq"], b["gain"], b["bandwidth"]) for b in bands)
    current_hash = hash(bands_tuple)
    
    # Only return false if hash matches AND we are not forcing update (via reset hash)
    if st.session_state.generic_hash == current_hash and st.session_state.generic_hash is not None:
        return False
    
    st.session_state.generic_hash = current_hash
    
    # --- FIXED: Use Assignment (=) for Generic Mode ---
    modified_fft = apply_gain_mask_to_fft(
        st.session_state.fft_data,
        st.session_state.full_freqs,
        bands,
        st.session_state.sample_rate,
        use_multiplication=False
    )
    
    equalized_full = cooley_tukey_ifft(modified_fft)
    equalized_audio = equalized_full[:len(st.session_state.audio_data)]
    
    update_output(equalized_audio)
    return True


def process_preset_eq(gains, mode):
    if st.session_state.fft_data is None:
        return False
    
    gains_tuple = tuple(gains)
    current_hash = hash(gains_tuple)
    
    if st.session_state.preset_hashes.get(mode) == current_hash:
        return False
    
    st.session_state.preset_hashes[mode] = current_hash
    
   # CHANGED: Only use linear subtraction for Human Voices Mode
    if mode == "Human Voices Mode" and PRECOMPUTED_DATA is not None:
        modified_fft = apply_linear_subtraction(st.session_state.fft_data, gains, PRECOMPUTED_DATA)
    else:
        sources = PRESETS.get(mode, {})
        preset_bands = []
        for i, s_name in enumerate(sources):
            g_val = gains[i] if i < len(gains) else 1.0
            for center, bw in sources[s_name]:
                preset_bands.append({"freq": center, "gain": g_val, "bandwidth": bw})
        
        # --- FIXED: Use Multiplication (*=) for Preset Mode to handle overlaps ---
        modified_fft = apply_gain_mask_to_fft(
            st.session_state.fft_data,
            st.session_state.full_freqs,
            preset_bands,
            st.session_state.sample_rate,
            use_multiplication=True
        )
    
    equalized_full = cooley_tukey_ifft(modified_fft)
    equalized_audio = equalized_full[:len(st.session_state.audio_data)]
    
    update_output(equalized_audio)
    
    return True


def run_ai_processing(gains, mode):
    
    if mode not in ['Musical Instruments Mode', 'Animal Sounds Mode']:
        return


    if st.session_state.ai_server_status is None:
        st.session_state.ai_server_status = AIAudioProcessor.get_available_servers()

    server_key = 'spleeter' if mode == 'Musical Instruments Mode' else 'animal_separator'
    server_status = st.session_state.ai_server_status[server_key]

    if server_status['status'] != 'healthy':
        return

    sources = PRESETS.get(mode, {})
    
    # --- LOGIC BRANCHING ---
    
    # CASE A: Animal Sounds Mode (Supports Dictionary/Multi-mix)
    if mode == 'Animal Sounds Mode':
        adjustments = {}
        any_active = False
        
        # Collect all slider values into one dictionary
        for i, source_name in enumerate(sources):
            gain = gains[i] if i < len(gains) else 1.0
            adjustments[source_name] = gain
            # Check if at least one slider is modifying the sound
            if abs(gain - 1.0) > 0.01:
                any_active = True
        
        # If there are adjustments, send the whole dictionary
        if any_active:
            result = AIAudioProcessor.process_audio(
                audio_data=st.session_state.audio_data,
                sample_rate=st.session_state.sample_rate,
                mode=mode,
                classifier=adjustments, # <--- Passing Dict here
                loudness=None,
                progress_callback=None
            )
            if result is not None:
                ai_audio, ai_sr = result
                update_ai_output(ai_audio)

    # CASE B: Musical Instruments Mode (Spleeter - Single Stem Focus)
    else:
        # Keep original logic: find the first modified stem and process it
        for i, source_name in enumerate(sources):
            gain = gains[i] if i < len(gains) else 1.0
            if abs(gain - 1.0) > 0.01:
                result = AIAudioProcessor.process_audio(
                    audio_data=st.session_state.audio_data,
                    sample_rate=st.session_state.sample_rate,
                    mode=mode,
                    classifier=source_name, # <--- Passing String here
                    loudness=gain,
                    progress_callback=None
                )
                if result is not None:
                    ai_audio, ai_sr = result
                    update_ai_output(ai_audio)
                break



def sidebar_controls():
    st.subheader("Configuration")
    
    freq_scale = st.radio("Display", ["Linear FFT", "Audiogram (dB HL)"], index=0, key="freq_scale_radio")
    st.session_state.freq_scale = freq_scale
    
    mode_options = ["Generic Mode", "Musical Instruments Mode", "Animal Sounds Mode", "Human Voices Mode"]
    
    # Callback to reset output to input when mode changes
    def on_mode_change():
        # Check if we have audio loaded before trying to reset
        if st.session_state.get('input_audio_b64') is not None:
            # Copy Input data to Output data
            st.session_state.output_audio_b64 = st.session_state.input_audio_b64
            st.session_state.output_fft_data = st.session_state.fft_data
            st.session_state.output_fft_hash = st.session_state.input_fft_hash
            st.session_state.output_spec_data = st.session_state.input_spec_data
            
            # Clear AI data (hidden until processed)
            st.session_state.ai_audio_b64 = None
            st.session_state.ai_fft_data = None
            st.session_state.ai_fft_hash = None
            st.session_state.ai_spec_data = None
            
            # Reset processing triggers so new sliders work immediately
            st.session_state.generic_hash = None
            st.session_state.preset_hashes = {}
            
            # MODE SWITCH LOGIC
            # If switching TO Generic Mode, load the saved config AND APPLY IT
            if st.session_state.mode_select == 'Generic Mode':
                st.session_state.eq_bands = load_generic_config()
                # Apply immediately
                process_generic_eq(st.session_state.eq_bands)
            else:
                # If switching AWAY from Generic Mode, clear the bands for UI cleanliness
                st.session_state.eq_bands = []

    mode = st.selectbox("Mode:", mode_options, key="mode_select", on_change=on_mode_change)
    st.session_state.current_mode = mode
    
    uploaded_file = st.file_uploader("Upload Audio", type=['wav', 'mp3', 'flac'], key="audio_uploader")
    
    if mode in ['Musical Instruments Mode', 'Animal Sounds Mode']:
        if st.session_state.ai_server_status is None:
            st.session_state.ai_server_status = AIAudioProcessor.get_available_servers()
        server_key = 'spleeter' if mode == 'Musical Instruments Mode' else 'animal_separator'
        server_status = st.session_state.ai_server_status[server_key]
        if server_status['status'] == 'healthy':
            st.success("AI Server Online")
        else:
            st.warning("AI Server Offline - Using Preset Mode")
    
    return uploaded_file


@st.fragment
def generic_mode_sliders():
    # Modified top row: Add Band and Save Config buttons
    col_add, col_save, _ = st.columns([1, 1, 4])
    
    with col_add:
        if st.button("Add Band", key="add_band_btn"):
            st.session_state.eq_bands.append({
                "id": str(uuid.uuid4()),
                "freq": 1000,
                "gain": 1.0,
                "bandwidth": 200
            })
            
    with col_save:
        if st.button("Save Config", key="save_generic_btn"):
            if save_generic_config(st.session_state.eq_bands):
                st.toast("Configuration saved!", icon="💾")
            else:
                st.error("Failed to save configuration")
    
    # --- NEW CALLBACK FUNCTION ---
    # This runs BEFORE the fragment reruns, ensuring the deleted item is gone
    # from the list before the loop iterates to draw sliders.
    def delete_band_callback(band_id):
        # 1. Remove from state
        st.session_state.eq_bands = [b for b in st.session_state.eq_bands if b["id"] != band_id]
        # 2. Update audio immediately so the sound reflects the removal
        process_generic_eq(st.session_state.eq_bands)

    bands_changed = False
    
    for idx, band in enumerate(st.session_state.eq_bands):
        cols = st.columns([3, 3, 3, 1])
        freq = cols[0].slider(f"Freq (Hz)", 20, st.session_state.sample_rate // 2, int(band["freq"]), key=f"f_{band['id']}")
        gain = cols[1].slider(f"Gain", 0.0, 2.0, float(band["gain"]), key=f"g_{band['id']}")
        bw = cols[2].slider(f"Width", 10, 5000, int(band["bandwidth"]), key=f"b_{band['id']}")
        
        # Use on_click callback for instant removal
        cols[3].button("X", key=f"d_{band['id']}", on_click=delete_band_callback, args=(band["id"],))
        
        if band["freq"] != freq or band["gain"] != gain or band["bandwidth"] != bw:
            bands_changed = True
        band["freq"] = freq
        band["gain"] = gain
        band["bandwidth"] = bw
    
    # Only process normal changes (sliders) here. 
    # Deletions are handled by the callback above.
    if bands_changed:
        process_generic_eq(st.session_state.eq_bands)
        st.rerun() # <--- Forces output graphs and player to update immediately on slider drag


@st.fragment
def preset_mode_sliders(current_mode):
    if current_mode not in PRESETS:
        st.warning(f"No preset available for '{current_mode}'")
        return
    
    sources = PRESETS[current_mode]
    num_sources = len(sources)
    cols = st.columns(min(num_sources, 5))
    
    source_gains = []
    for i, source_name in enumerate(sources):
        with cols[i % len(cols)]:
            gain = st.slider(
                source_name.capitalize(),
                0.0, 2.0, 1.0, 0.05,
                key=f"preset_{current_mode}_{source_name}"
            )
            source_gains.append(gain)
    
    # Logic: If processing happens, we MUST rerun to update the layout
    # because this function is a fragment.
    if process_preset_eq(source_gains, current_mode):
        run_ai_processing(source_gains, current_mode)
        st.rerun() # <--- Added this to fix layout issues

@st.fragment
def render_input_player():
    st.markdown("#### Input Signal")
    if st.session_state.input_audio_b64:
        audio_hash = hash(st.session_state.input_audio_b64[:100])
        player_html = create_audio_player_html(audio_hash, st.session_state.input_audio_b64, "input")
        components.html(player_html, height=200, scrolling=False)


@st.fragment
def render_output_player():
    st.markdown("#### Equalizer Output")
    if st.session_state.output_audio_b64:
        audio_hash = hash(st.session_state.output_audio_b64[:100])
        player_html = create_audio_player_html(audio_hash, st.session_state.output_audio_b64, "output")
        components.html(player_html, height=200, scrolling=False)


@st.fragment
def render_ai_player():
    st.markdown("#### AI Output")
    if st.session_state.ai_audio_b64:
        audio_hash = hash(st.session_state.ai_audio_b64[:100])
        player_html = create_audio_player_html(audio_hash, st.session_state.ai_audio_b64, "ai")
        components.html(player_html, height=200, scrolling=False)
    else:
        st.caption("Adjust sliders to generate AI output")


@st.fragment
def render_input_graph():
    freq_scale = st.session_state.freq_scale
    sample_rate = st.session_state.sample_rate
    fft_data = st.session_state.fft_data
    fft_hash = st.session_state.input_fft_hash
    
    if fft_data is not None and sample_rate:
        if freq_scale == "Linear FFT":
            fig = create_fft_fig(fft_hash, fft_data.tobytes(), sample_rate, "Input FFT", 'steelblue')
            st.plotly_chart(fig, width='content', key="input_fft")
        else:
            fig = create_audiogram_fig(fft_hash, fft_data.tobytes(), sample_rate, "Input Audiogram", 'steelblue')
            st.plotly_chart(fig, width='content', key="input_audiogram")


@st.fragment
def render_output_graph():
    freq_scale = st.session_state.freq_scale
    sample_rate = st.session_state.sample_rate
    fft_data = st.session_state.output_fft_data
    fft_hash = st.session_state.output_fft_hash
    
    if fft_data is not None and sample_rate:
        if freq_scale == "Linear FFT":
            fig = create_fft_fig(fft_hash, fft_data.tobytes(), sample_rate, "Output FFT", 'orange')
            st.plotly_chart(fig, width='content', key="output_fft")
        else:
            fig = create_audiogram_fig(fft_hash, fft_data.tobytes(), sample_rate, "Output Audiogram", 'orange')
            st.plotly_chart(fig, width='content', key="output_audiogram")


@st.fragment
def render_ai_graph():
    freq_scale = st.session_state.freq_scale
    sample_rate = st.session_state.sample_rate
    fft_data = st.session_state.ai_fft_data
    fft_hash = st.session_state.ai_fft_hash
    
    if fft_data is not None and sample_rate:
        if freq_scale == "Linear FFT":
            fig = create_fft_fig(fft_hash, fft_data.tobytes(), sample_rate, "AI Output FFT", 'green')
            st.plotly_chart(fig, width='content', key="ai_fft")
        else:
            fig = create_audiogram_fig(fft_hash, fft_data.tobytes(), sample_rate, "AI Output Audiogram", 'green')
            st.plotly_chart(fig, width='content', key="ai_audiogram")
    else:
        st.empty()


@st.fragment
def render_input_spectrogram():
    sample_rate = st.session_state.sample_rate
    spec_data = st.session_state.input_spec_data
    
    if spec_data is not None and sample_rate:
        spec_hash = hash(spec_data.tobytes()[:1000])
        fig = create_spectrogram_fig(spec_hash, spec_data.tobytes(), spec_data.shape, sample_rate, 512, "Input Spectrogram")
        st.plotly_chart(fig, width='content', key="input_spec")


@st.fragment
def render_output_spectrogram():
    sample_rate = st.session_state.sample_rate
    spec_data = st.session_state.output_spec_data
    
    if spec_data is not None and sample_rate:
        spec_hash = hash(spec_data.tobytes()[:1000])
        fig = create_spectrogram_fig(spec_hash, spec_data.tobytes(), spec_data.shape, sample_rate, 512, "Output Spectrogram")
        st.plotly_chart(fig, width='content', key="output_spec")


@st.fragment
def render_ai_spectrogram():
    sample_rate = st.session_state.sample_rate
    spec_data = st.session_state.ai_spec_data
    
    if spec_data is not None and sample_rate:
        spec_hash = hash(spec_data.tobytes()[:1000])
        fig = create_spectrogram_fig(spec_hash, spec_data.tobytes(), spec_data.shape, sample_rate, 512, "AI Spectrogram")
        st.plotly_chart(fig, width='content', key="ai_spec")
    else:
        st.empty()


st.markdown('<h1 class="main-header">Signal Equalizer</h1>', unsafe_allow_html=True)

with st.sidebar:
    uploaded_file = sidebar_controls()

if uploaded_file is not None:
    with st.spinner("Analyzing audio..."):
        try:
            process_new_audio(uploaded_file)
        except Exception as e:
            st.error(f"Error processing audio: {e}")


if st.session_state.audio_data is not None:
    st.markdown("---")
    
    current_mode = st.session_state.current_mode
    st.markdown(f"#### Equalizer Controls - {current_mode}")
    
    if current_mode == "Generic Mode":
        generic_mode_sliders()
    else:
        preset_mode_sliders(current_mode)
    
    st.markdown("---")
    
    has_output = st.session_state.output_audio_b64 is not None
    has_ai = st.session_state.ai_audio_b64 is not None
    is_ai_mode = current_mode in ['Musical Instruments Mode', 'Animal Sounds Mode']
    
    if is_ai_mode or has_ai:
        col1, col2, col3 = st.columns(3)
    elif has_output:
        col1, col2 = st.columns(2)
        col3 = None
    else:
        col1 = st.container()
        col2 = None
        col3 = None
    
    with col1:
        render_input_player()
        render_input_graph()
    
    if col2 is not None:
        with col2:
            render_output_player()
            render_output_graph()
    
    if col3 is not None:
        with col3:
            render_ai_player()
            render_ai_graph()
    
    st.markdown("---")
    
    with st.expander("Spectrograms", expanded=True):
        if is_ai_mode or has_ai:
            spec_cols = st.columns(3)
        elif has_output:
            spec_cols = st.columns(2)
        else:
            spec_cols = [st.container()]
        
        with spec_cols[0]:
            render_input_spectrogram()
        
        if has_output and len(spec_cols) > 1:
            with spec_cols[1]:
                render_output_spectrogram()
        
        if (is_ai_mode or has_ai) and len(spec_cols) > 2:
            with spec_cols[2]:
                render_ai_spectrogram()

else:
    st.info("Upload an audio file to get started")