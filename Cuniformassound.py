import streamlit as st
import numpy as np
from scipy.io.wavfile import write

st.set_page_config(layout="wide", page_title="The Fuente Magna Bowl Simulation")

def generate_tone(frequency, duration_s, sample_rate=44100):
    t = np.linspace(0., duration_s, int(sample_rate * duration_s), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.7
    data = amplitude * np.sin(2. * np.pi * frequency * t)
    return data.astype(np.int16)

def apply_geometric_lens_effect(pulse_wave, num_layers, base_freq, sample_rate=44100):
    max_delay_seconds = (1.0 / base_freq) * (num_layers * 2)
    max_delay_samples = int(max_delay_seconds * sample_rate)
    final_length = len(pulse_wave) + max_delay_samples
    
    final_wave = np.zeros(final_length, dtype=np.float64)
    final_wave[:len(pulse_wave)] += pulse_wave.astype(np.float64)

    for i in range(1, num_layers + 1):
        delay_seconds = (1.0 / base_freq) * (i * 2)
        delay_samples = int(delay_seconds * sample_rate)
        echo_amplitude = 0.8 / (i + 1)
        
        start_index = delay_samples
        end_index = delay_samples + len(pulse_wave)
        
        if end_index <= final_length:
            final_wave[start_index:end_index] += pulse_wave.astype(np.float64) * echo_amplitude

    max_amp = np.max(np.abs(final_wave))
    if max_amp > 0:
        final_wave = (final_wave / max_amp) * np.iinfo(np.int16).max * 0.9
        
    return final_wave.astype(np.int16)

st.title("Step 1: Audio Simulation of the Firing Process")
# ... (rest of the UI from the previous turn) ...
FROG_FREQ = 136.1
CHARGE_DURATION_S = 4
PULSE_DURATION_S = 0.5
NUM_SQUARES = 4

if st.button("▶️ RUN FULL AUDIO SIMULATION"):
    # (The logic from the previous turn goes here, it is correct with the functions above)
    pass
