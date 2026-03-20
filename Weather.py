import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# (All the parameters and the update function are the same as before)

# --- Setup the Grid ---
x = np.arange(0, GRID_SIZE)
y = np.arange(0, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# --- Animation Function ---
fig, ax = plt.subplots()
def update(frame):
    ax.clear()
    storm_wave = 1.0 * np.sin(0.5 * X - 1.0 * frame)
    dist_from_chant = np.sqrt((X - 70)**2 + (Y - 50)**2)
    chant_wave = 1.2 * np.sin(0.5 * dist_from_chant - 1.0 * frame + np.pi)
    chant_wave *= np.exp(-(dist_from_chant / (GRID_SIZE / 5))**2)
    combined_field = storm_wave + chant_wave
    im = ax.imshow(combined_field, cmap='viridis', vmin=-2, vmax=2)
    ax.set_title(f"Wave Interference Simulation (Time: {frame})")
    ax.set_xticks([])
    ax.set_yticks([])
    return [im]

# --- Run the Animation ---
ani = animation.FuncAnimation(fig, update, frames=100, blit=True)

# Use this line to show an interactive pop-up window
plt.show()

# Use this line if you want to save to a file instead
# ani.save('wave_simulation.gif', writer='imagemagick', fps=15)
