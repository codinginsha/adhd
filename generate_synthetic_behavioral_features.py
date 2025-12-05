import numpy as np
import pandas as pd

def generate_behavioral_features(n_samples=500, seed=42, out_file='behavioral_features.csv'):
    np.random.seed(seed)
    gaze_contact = np.clip(np.random.normal(0.6, 0.15, n_samples), 0, 1)
    head_rotation = np.clip(np.random.normal(10, 6, n_samples), -90, 90)
    smile_score = np.clip(np.random.beta(2,5, n_samples), 0, 1)
    joint_attention = np.clip(np.random.normal(0.3, 0.2, n_samples), 0, 1)
    prosody_energy = np.clip(np.random.normal(0.5, 0.2, n_samples), 0, 2)
    risk_score = ( (1 - gaze_contact)*0.4 + (np.abs(head_rotation)/90)*0.2 + (1 - smile_score)*0.2 + (1 - joint_attention)*0.2 )
    df = pd.DataFrame({
        'gaze_contact': gaze_contact,
        'head_rotation': head_rotation,
        'smile_score': smile_score,
        'joint_attention': joint_attention,
        'prosody_energy': prosody_energy,
        'risk_score': risk_score
    })
    df.to_csv(out_file, index=False)
    print(f'Behavioral features saved to {out_file}')

if __name__ == '__main__':
    generate_behavioral_features(out_file='behavioral_features.csv')
