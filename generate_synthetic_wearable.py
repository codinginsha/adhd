import numpy as np
import pandas as pd

def generate_wearable_timeseries(n_subjects=20, seq_len=300, seed=1, out_file='wearable_timeseries.csv'):
    np.random.seed(seed)
    rows = []
    for sid in range(n_subjects):
        hr_base = np.random.uniform(60, 100)
        eda_base = np.random.uniform(0.1, 3.0)
        for t in range(seq_len):
            hr = hr_base + np.random.normal(0, 3) + 5*np.sin(2*np.pi*t/60)
            eda = max(0, eda_base + np.random.normal(0, 0.2) + 0.2*np.sin(2*np.pi*t/90))
            accel = np.abs(np.random.normal(0, 0.5)) + 0.5*(np.random.rand()<0.02)
            prob_inattentive = 0.1 + 0.4*(np.abs(np.sin(2*np.pi*t/300)))
            attention = 0 if (np.random.rand() < prob_inattentive) else 1
            rows.append([sid, t, hr, eda, accel, attention])
    df = pd.DataFrame(rows, columns=['subject','t','hr','eda','accel','attention'])
    df.to_csv(out_file, index=False)
    print(f'Wearable timeseries saved to {out_file}')

if __name__ == '__main__':
    generate_wearable_timeseries(out_file='wearable_timeseries.csv')
