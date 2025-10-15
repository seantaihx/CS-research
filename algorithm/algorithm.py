import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy import signal

# 1. Generate the independent source signals
np.random.seed(42)
n_samples = 2000
time = np.linspace(0, 20, n_samples)

s1_raw = np.sin(2 * np.pi * time)  # Sinusoidal signal
s2_raw = signal.sawtooth(3 * np.pi * time)  # Sawtooth signal
s3_raw = np.random.randn(n_samples) * 0.5  # Random noise signal

# Define activity windows for each series, including a 3-series overlap
dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='min')
print(dates)
s1_active = (dates >= '2023-01-01 00:00') & (dates <= '2023-01-01 22:00')
s2_active = (dates >= '2023-01-01 10:00') & (dates <= '2023-01-02 05:00')
s3_active = (dates >= '2023-01-02 00:00') & (dates <= '2023-01-02 10:00')  

s1 = pd.Series(np.zeros(n_samples), index=dates)
s2 = pd.Series(np.zeros(n_samples), index=dates)
s3 = pd.Series(np.zeros(n_samples), index=dates)

s1[s1_active] = s1_raw[s1_active]
s2[s2_active] = s2_raw[s2_active]
s3[s3_active] = s3_raw[s3_active]

# Generate the total mixed signal
total_signal = s1 + s2 + s3

# 2. Hybrid separation
s1_separated = pd.Series(np.zeros(n_samples), index=dates)
s2_separated = pd.Series(np.zeros(n_samples), index=dates)
s3_separated = pd.Series(np.zeros(n_samples), index=dates)

# Find all unique intervals based on activity
intervals = pd.Series(0, index = dates)
intervals[(s1_active) & ~(s2_active) & ~(s3_active)] = 1
intervals[~(s1_active) & (s2_active) & ~(s3_active)] = 2
intervals[~(s1_active) & ~(s2_active) & (s3_active)] = 3
intervals[(s1_active) & (s2_active) & ~(s3_active)] = 4
intervals[~(s1_active) & (s2_active) & (s3_active)] = 5
intervals[(s1_active) & (s2_active) & (s3_active)] = 6

# Process each interval
for interval_type, group in intervals.groupby(intervals):
    if interval_type == 1:
        s1_separated[group.index] = total_signal[group.index]
    elif interval_type == 2:
        s2_separated[group.index] = total_signal[group.index]
    elif interval_type == 3:
        s3_separated[group.index] = total_signal[group.index]
    elif interval_type == 4:
        mixed_segment = total_signal[group.index]
        mixed_matrix = pd.DataFrame({'lag_0': mixed_segment,
'lag_1': mixed_segment.shift(1)}).dropna()
        ica = FastICA(n_components=2, whiten='arbitrary-variance', random_state=42)
        separated_segment = ica.fit_transform(mixed_matrix.values)
        s1_separated.loc[mixed_matrix.index] = separated_segment[:, 0]
        s2_separated.loc[mixed_matrix.index] = separated_segment[:, 1]
    elif interval_type == 5:
        mixed_segment = total_signal[group.index]
        mixed_matrix = pd.DataFrame({'lag_0': mixed_segment,
'lag_1': mixed_segment.shift(1)}).dropna()
        ica = FastICA(n_components=2, whiten='arbitrary-variance', random_state=42)
        separated_segment = ica.fit_transform(mixed_matrix.values)
        s2_separated.loc[mixed_matrix.index] = separated_segment[:, 0]
        s3_separated.loc[mixed_matrix.index] = separated_segment[:, 1]
    elif interval_type == 6:
        mixed_segment = total_signal[group.index]
        mixed_matrix = pd.DataFrame({'lag_0': mixed_segment,
'lag_1': mixed_segment.shift(1),
'lag_2': mixed_segment.shift(2)}).dropna()
        ica = FastICA(n_components=3, whiten='arbitrary-variance', random_state=42)
        separated_segment = ica.fit_transform(mixed_matrix.values)

        # Match separated components to original signals based on correlation
        # This part requires more advanced logic or manual inspection
        s1_separated.loc[mixed_matrix.index] = separated_segment[:, 0]
        s2_separated.loc[mixed_matrix.index] = separated_segment[:, 1]
        s3_separated.loc[mixed_matrix.index] = separated_segment[:, 2]

    # Re-normalize ICA outputs to match original scales
    # ... (same as before)

    # 3. Visualize the results
plt.figure(figsize=(15, 12))
plt.subplot(4, 1, 1)
plt.title('True Sources')
plt.plot(s1, label='Source 1 (Sinusoid)')
plt.plot(s2, label='Source 2 (Sawtooth)')
plt.plot(s3, label='Source 3 (Noise)')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.title('Total Mixed Signal')
plt.plot(total_signal, color='black', label='Mixed Signal')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.title('Hybrid Separated Sources (Adjusted Scales)')
plt.plot(s1_separated, label='Separated Source 1', linestyle='--')
plt.plot(s2_separated, label='Separated Source 2', linestyle='--')
plt.plot(s3_separated, label='Separated Source 3', linestyle='--')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.savefig('hybrid_separation_results.png')
