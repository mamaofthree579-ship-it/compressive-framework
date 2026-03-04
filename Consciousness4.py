import numpy as np

# time
dt = 0.01
t = np.arange(0, 60, dt)

# natural frequencies (Hz)
f_gut = 0.05 # slow gut rhythm
f_heart = 1.2 # baseline heart
f_brain_voice = 5.0 # inner voice ~theta
f_brain_visual = 10.0 # visual ~alpha

# emotion modulation (e.g., fear speeds heart)
def heart_freq(emotion):
    return f_heart + {'calm':0, 'fear':0.3, 'joy':0.1}[emotion]

# coupling strength
K = 0.8
gut_drive = 0.2 # occasional gut burst

def simulate(brain_mode='both', emotion='fear'):
    # choose brain freq(s)
    fb = [f_brain_voice] if brain_mode=='voice' else \
         [f_brain_visual] if brain_mode=='visual' else \
         [f_brain_voice, f_brain_visual]

    phases = np.zeros((len(fb)+2, len(t))) # gut, heart, brain(s)
    for i in range(1, len(t)):
        # gut gets a burst at t≈20s
        drive = gut_drive*np.sin(2*np.pi*0.2*t[i]) if 19<t[i]<21 else 0
        # update gut
        phases[0,i] = phases[0,i-1] + dt*(2*np.pi*f_gut + K*np.sin(phases[1,i-1]-phases[0,i-1]) + drive)
        # update heart
        phases[1,i] = phases[1,i-1] + dt*(2*np.pi*heart_freq(emotion) + K*np.sin(phases[0,i-1]-phases[1,i-1]))
        # update brain(s)
        for j, fbj in enumerate(fb):
            idx = 2+j
            phases[idx,i] = phases[idx,i-1] + dt*(2*np.pi*fbj + K*np.sin(phases[1,i-1]-phases[idx,i-1]))

    # locking: phase diff between heart and each brain stays <0.5 rad for ≥3 cycles
    lock = {}
    for j, mode in enumerate(['voice','visual'][:len(fb)]):
        diff = np.angle(np.exp(1j*(phases[1]-phases[2+j])))
        lock[mode] = np.any([
            np.all(np.abs(diff[k:k+int(3/(heart_freq(emotion)) / dt)])<0.5
            for k in range(len(diff)-int(3/(heart_freq(emotion)) / dt))
        ])
    return lock

# example run
print(simulate(brain_mode='both', emotion='fear'))
