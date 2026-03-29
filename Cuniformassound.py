import streamlit as st
import numpy as np
from scipy.io.wavfile import write

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Cuneiform Sound Synthesizer")

# --- Functions ---

def generate_tone(frequency, duration_s, sample_rate=44100):
    """Generates a pure sine wave."""
    t = np.linspace(0., duration_s, int(sample_rate * duration_s), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2. * np.pi * frequency * t)
    return data.astype(np.int16)

def apply_envelope(data, attack_s, decay_s, sustain_level, sample_rate=44100):
    """Applies an ADSR-like envelope to a tone. More robust version."""
    total_samples = len(data)
    attack_samples = int(attack_s * sample_rate)
    decay_samples = int(decay_s * sample_rate)
    
    # Ensure attack and decay don't exceed total length
    if attack_samples > total_samples:
        attack_samples = total_samples
        decay_samples = 0
    elif attack_samples + decay_samples > total_samples:
        decay_samples = total_samples - attack_samples

    sustain_samples = total_samples - attack_samples - decay_samples

    # Create envelope segments
    attack = np.linspace(0, 1, attack_samples) if attack_samples > 0 else np.array([])
    decay = np.linspace(1, sustain_level, decay_samples) if decay_samples > 0 else np.array([])
    sustain = np.full(sustain_samples, sustain_level) if sustain_samples > 0 else np.array([])
    
    # Combine segments to form the final envelope
    envelope = np.concatenate((attack, decay, sustain))
    
    return (data * envelope).astype(np.int16)

def synthesize_script(script, params, sample_rate=44100):
    """Translates a cuneiform script into a waveform."""
    final_waveform = np.array([], dtype=np.int16)
    characters = script.split(' ')
    
    for char in characters:
        if not char:
            continue
            
        base_freq = params["base_freq"]
        duration_s = params["duration_ms"] / 1000.0
        
        # 1. Generate Base Tone (from '|')
        if '|' in char:
            char_wave = generate_tone(base_freq, duration_s, sample_rate)
        else:
            char_wave = np.zeros(int(duration_s * sample_rate), dtype=np.int16)

        # 2. Add Harmonic (from '-')
        if '-' in char:
            harmonic_freq = base_freq * params["harmonic_multiplier"]
            harmonic_wave = generate_tone(harmonic_freq, duration_s, sample_rate)
            char_wave = (char_wave * 0.7 + harmonic_wave * 0.5).astype(np.int16)

        # 3. Apply Envelope (from '>')
        if '>' in char: # Sharp attack
            attack_s = 0.005 
            decay_s = params["sharp_decay_s"]
            sustain_level = 0.1
        else: # Normal attack
            attack_s = 0.05
            decay_s = duration_s * 0.4
            sustain_level = 0.7

        char_wave = apply_envelope(char_wave, attack_s, decay_s, sustain_level, sample_rate)
        
        # Add a small silence between characters
        silence = np.zeros(int(sample_rate * 0.05), dtype=np.int16)
        final_waveform = np.concatenate((final_waveform, char_wave, silence))
        
    return final_waveform

# --- Streamlit UI ---

st.title("The Waveform Hypothesis: Cuneiform Sound Synthesizer")
st.markdown(
    "This tool tests the hypothesis that cuneiform-like scripts could be visual representations of sound waves. "
    "By assigning sonic properties to basic geometric components, we can synthesize the audio these symbols might encode."
)

st.sidebar.title("Controls")
st.sidebar.markdown("### 1. The Cuneiform Script")
legend_text = """
**Legend:**
- `|` = Base Tone (Vertical Stroke)
- `-` = Harmonic (Horizontal Stroke)
- `>` = Sharp Attack (Wedge)

Combine them into 'characters' and separate each character with a space.
"""
st.sidebar.markdown(legend_text)

script_input = st.sidebar.text_input("Enter Script:", value="|> |-| |")

st.sidebar.markdown("### 2. Sonic Parameters")
params = {}
params["base_freq"] = st.sidebar.slider("Base Frequency for '|' (Hz)", 100, 800, 220)
params["harmonic_multiplier"] = st.sidebar.slider("Harmonic Multiplier for '-'", 1.0, 4.0, 1.5, step=0.1)
params["duration_ms"] = st.sidebar.slider("Duration per Symbol (ms)", 100, 1000, 400)
params["sharp_decay_s"] = st.sidebar.slider("Decay for Sharp Attack '>' (s)", 0.05, 1.0, 0.3, step=0.05)

# --- Main App Logic ---

if st.button("Generate Sound from Script"):
    st.subheader("Synthesized Audio & Waveform")
    
    if not script_input.strip():
        st.warning("Please enter a script in the sidebar.")
    else:
        with st.spinner("Synthesizing waveform..."):
            # Generate the waveform
            synthesized_wave = synthesize_script(script_input, params)
            
            # Save as a temporary WAV file
            wav_filename = "cuneiform_synthesis.wav"
            write(wav_filename, 44100, synthesized_wave)
            
            # Display Audio Player
            st.audio(wav_filename)
            
            # Display Waveform Chart
            st.markdown("##### Waveform Visualization")
            # Downsample for faster plotting if waveform is long
            plot_data = synthesized_wave
            if len(plot_data) > 100000:
                 plot_data = synthesized_wave[::10]
            st.line_chart(plot_data)
            
            st.success(f"Successfully generated sound for script: `{script_input}`")

st.markdown("---")
st.markdown(
    """#### How to Experiment:
1. **Test Single Components:** Start with just `|`, then `-`, then `>`. Hear what each one does.
2. **Build Complexity:** Create characters like `|-` (a tone with its harmonic) and `|>` (a percussive tone).
3. **Create a Rhythm:** Use different combinations in a sequence, like `|> | - |>`. Listen for the cadence and rhythm that emerges from the visual symbols."""
)
