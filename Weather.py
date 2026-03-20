import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters for a single frame ---
# This section defines all the necessary variables.
# The error happens if this part is missing.
GRID_SIZE = 100
FRAME = 50 

# --- Setup and Calculation ---
# The line below is where the error occurred because GRID_SIZE wasn't defined yet.
x = np.arange(0, GRID_SIZE)
y = np.arange(0, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# Calculate the waves at this single moment in time
storm_wave = 1.0 * np.sin(0.5 * X - 1.0 * FRAME)
dist_from_chant = np.sqrt((X - 70)**2 + (Y - 50)**2)
chant_wave = 1.2 * np.sin(0.5 * dist_from_chant - 1.0 * FRAME + np.pi)
chant_wave *= np.exp(-(dist_from_chant / (GRID_SIZE / 5))**2)
combined_field = storm_wave + chant_wave

# --- Plot the single image ---
fig, ax = plt.subplots()
im = ax.imshow(combined_field, cmap='viridis', vmin=-2, vmax=2)
ax.set_title(f"Wave Interference (Single Frame)")
ax.set_xticks([])
ax.set_yticks([])

# This line will try to open an interactive window to show the plot
plt.show()

# You can uncomment the line below to save the image as a file instead
# plt.savefig('static_wave_frame.png')
