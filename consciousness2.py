if uploaded:
    # after loading eeg and t
    if t[0] > t[-1]: # flip if descending
        t = t[::-1]; eeg = eeg[::-1]
    eeg = eeg - np.mean(eeg) # detrend
