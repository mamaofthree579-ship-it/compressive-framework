import matplotlib
# This line is the potential fix. It MUST be before importing pyplot.
# It tells matplotlib to use a specific, reliable graphics engine called 'TkAgg'.
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation Parameters ---
GRID_SIZE = 100
TIME_STEPS = 120 # Increased steps for a longer animation

# Storm parameters
STORM_FREQ = 0.5 
STORM_AMP = 1.0 
STORM_SPEED = 1.0

# Intervention ("Chant") parameters
CHANT_FREQ = 0.5 
CHANT_AMP = 1.2 
CHANT_POS = (70, 50) 
CHANT_PHASE = np.pi # Set to pi (180 degrees) for destructive interference

# --- Setup the Grid ---
x = np.arange(0, GRID_SIZE)
y = np.arange(0, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# --- Animation Setup ---
fig, ax = plt.subplots()

# --- Update Function ---
# This is the core of the simulation, called for each frame.
def update(frame):
    ax.clear()

    # 1. Propagate the storm wave
    storm_wave = STORM_AMP * np.sin(STORM_FREQ * X - STORM_SPEED * frame)

    # 2. Generate the intervention wave from a point
    dist_from_chant = np.sqrt((X - CHANT_POS[0])**2 + (Y - CHANT_POS[1])**2)
    chant_wave = CHANT_AMP * np.sin(CHANT_FREQ * dist_from_chant - STORM_SPEED * frame + CHANT_PHASE)
    
    # Localize the chant wave so it fades with distance
    chant_wave *= np.exp(-(dist_from_chant / (GRID_SIZE / 5))**2)

    # 3. Combine the waves to simulate interference
    combined_field = storm_wave + chant_wave

    # 4. Draw the result
    im = ax.imshow(combined_field, cmap='viridis', vmin=-2, vmax=2, animated=True)
    ax.set_title(f"Wave Interference Simulation (Time: {frame})")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Return the artist object that was drawn
    return [im]

# --- Run the Animation ---
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=TIME_STEPS, 
    interval=50, # Delay between frames in milliseconds
    blit=True
)

# This command should now open an interactive window and show the animation
plt.show()
