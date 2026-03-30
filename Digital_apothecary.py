import streamlit as st
import numpy as np
from scipy.io.wavfile import write

st.set_page_config(layout="wide", page_title="Digital Apothecary")

# --- CORE FUNCTIONS ---
def generate_base_wave(frequency, duration, sample_rate=44100):
    """Generates the fundamental carrier wave."""
    t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2. * np.pi * frequency * t)
    return data.astype(np.int16)

def apply_geometric_signature(wave, geometry, base_freq, sample_rate=44100):
    """Modulates the wave with a geometric signature using harmonic echoes."""
    st.write(f"Applying signature: {geometry}")
    output_wave = np.copy(wave).astype(np.float64)
    num_layers = 6 # Layers of complexity

    for i in range(1, num_layers + 1):
        echo_amplitude = 0.6 / i
        delay_samples = 0

        # Each geometry creates a different harmonic pattern
        if geometry == "Concentric Circles (Expansion)":
            # Smooth, evenly spaced echoes - like ripples
            delay_seconds = (1.0 / base_freq) * (i * 2)
            delay_samples = int(delay_seconds * sample_rate)
        elif geometry == "Spiral (Growth/Contraction)":
            # Echoes get progressively faster or slower (golden ratio)
            phi = 1.61803398875
            delay_seconds = (1.0 / base_freq) * (phi**i)
            delay_samples = int(delay_seconds * sample_rate * 0.5) # scaled
        elif geometry == "Hexagon (Structure/Stability)":
            # Echoes based on the number 6, creating stable chords
            # Using intervals of a major third and a fifth to create a major chord feel
            intervals = [4/3, 3/2] # Perfect fourth, perfect fifth
            interval = intervals[(i-1) % len(intervals)]
            delay_seconds = (1.0 / (base_freq * interval)) * i
            delay_samples = int(delay_seconds * sample_rate)

        # Apply echo if it fits within the array bounds
        if delay_samples > 0 and delay_samples < len(output_wave):
            echo_wave = np.roll(wave, delay_samples) * echo_amplitude
            output_wave += echo_wave

    # Normalize
    max_amp = np.max(np.abs(output_wave))
    if max_amp > 0:
        output_wave = (output_wave / max_amp) * np.iinfo(np.int16).max * 0.9

    return output_wave.astype(np.int16)

# --- USER INTERFACE ---
st.title("The Digital Apothecary v1.0")
st.markdown("Generate custom vibrational fields by combining sacred geometry and fundamental frequencies.")

st.sidebar.header("Field Parameters")

# Geometry Selection
geom_choice = st.sidebar.selectbox(
    "1. Select a Core Geometry:",
    ("Concentric Circles (Expansion)", "Spiral (Growth/Contraction)", "Hexagon (Structure/Stability)")
)

# Frequency Selection
freq_map = {
    "136.1 Hz (OM / The Earth Year)": 136.1,
    "432 Hz (Universal Harmony)": 432.0,
    "528 Hz (Solfeggio 'Miracle' Tone)": 528.0
}
freq_choice_name = st.sidebar.selectbox(
    "2. Select a Base Frequency:",
    list(freq_map.keys())
)
freq_choice_hz = freq_map[freq_choice_name]

# Duration
duration_s = st.sidebar.slider("3. Set Duration (seconds):", 5, 30, 10)

if st.button("🌿 GENERATE SONIC ELIXIR"):
    with st.spinner("Brewing your elixir... This may take a moment."):
        # 1. Generate the base carrier wave
        base_wave = generate_base_wave(freq_choice_hz, duration_s)

        # 2. Modulate it with the chosen geometric signature
        final_elixir = apply_geometric_signature(base_wave, geom_choice, freq_choice_hz)

        # 3. Save and provide for download
        filename = f"elixir_{geom_choice.split(' ')[0]}_{int(freq_choice_hz)}hz.wav"
        write(filename, 44100, final_elixir)

        st.subheader("Your Sonic Elixir is Ready")
        st.audio(filename)
        st.download_button(
            label="Download.WAV File",
            data=open(filename, "rb"),
            file_name=filename,
            mime="audio/wav"
        )
        st.line_chart(final_elixir[::50]) # Show a preview of the waveform

st.markdown("---")
st.header("How to Use This Tool for Path 2")
st.markdown("""
By running these simulations, you are training your intuition. Listen carefully to the 'texture' of each sound:
- **Circles:** Feel smooth, radiating, and enveloping.
- **Spirals:** Sound like they are moving, evolving, 'swooping'.
- **Hexagons:** Feel stable, structured, almost crystalline and chord-like.

As we move to Path 2, we will search the archaeological record. When you see a spiral petroglyph at Newgrange in Ireland, or a hexagonal pattern on a Roman mosaic floor, you will already have a pre-conceived notion of its *energetic function*. You won't just see a picture; you'll hear a sound.

This apothecary is our training ground. Let's begin the next phase of our work.
""")
