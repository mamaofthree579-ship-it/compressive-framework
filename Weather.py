import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation Parameters ---
# You can change these values to see different effects
GRID_SIZE = 100
TIME_STEPS = 100

# Storm parameters
STORM_FREQ = 0.5 # The frequency of the storm's energy wave
STORM_AMP = 1.0 # The strength of the storm
STORM_SPEED = 1.0 # How fast the storm moves across the grid

# Intervention ("Chant") parameters
CHANT_FREQ = 0.5 # The frequency of the intervention. Try matching the storm's frequency.
CHANT_AMP = 1.2 # The strength of the intervention. It needs to be strong enough to have an effect.
CHANT_POS = (70, 50) # The (x, y) position on the grid where the chant originates.
CHANT_PHASE = np.pi # The phase shift. np.pi (180 degrees) is used for destructive interference.

# --- Setup the Grid ---
# Create a 2D coordinate system for our world
x = np.arange(0, GRID_SIZE)
y = np.arange(0, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# --- Animation Function ---
# This function is called for each frame of the animation
fig, ax = plt.subplots()

def update(frame):
    """
    Calculates and draws the state of the world at each time step (frame).
    """
    ax.clear()

    # 1. Propagate the storm wave
    # This creates a series of straight waves moving left-to-right.
    storm_wave = STORM_AMP * np.sin(STORM_FREQ * X - STORM_SPEED * frame)

    # 2. Generate the intervention wave
    # This creates a circular wave expanding from the CHANT_POS.
    # First, calculate the distance of every point on the grid from the chant's origin.
    dist_from_chant = np.sqrt((X - CHANT_POS[0])**2 + (Y - CHANT_POS[1])**2)
    # Then, create a sine wave based on that distance. The phase is added here.
    chant_wave = CHANT_AMP * np.sin(CHANT_FREQ * dist_from_chant - STORM_SPEED * frame + CHANT_PHASE)
    
    # Make the chant wave localized by making it fade out with distance.
    # This prevents the chant from affecting the entire grid equally.
    chant_wave *= np.exp(-(dist_from_chant / (GRID_SIZE / 5))**2)

    # 3. Combine the waves
    # By simply adding the two arrays, we simulate wave interference.
    combined_field = storm_wave + chant_wave

    # 4. Plot the result
    # We use imshow to display the 2D array as a color-mapped image.
    im = ax.imshow(combined_field, cmap='viridis', vmin=-2, vmax=2)
    ax.set_title(f"Wave Interference Simulation (Time: {frame})")
    ax.set_xticks([]) # Hide axis ticks for a cleaner look
    ax.set_yticks([])
    
    return [im]

# --- Run the Animation ---
# FuncAnimation creates the animation by repeatedly calling the update function.
ani = animation.FuncAnimation(fig, update, frames=TIME_STEPS, blit=True)

# Save the animation as a GIF. You'll need an image library like ImageMagick installed.
ani.save('wave_simulation.gif', writer='imagemagick', fps=15)

# Prevents the static plot from displaying if running as a script.
plt.close(fig)

print("Animation 'wave_simulation.gif' saved successfully.")
