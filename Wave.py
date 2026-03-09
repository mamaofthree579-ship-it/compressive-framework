import numpy as np
from lal import SimInspiralFD
# generate h+, hx, save h = h+ as array
np.save("h_lal.npy", h.real)
