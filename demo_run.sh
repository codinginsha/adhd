#!/bin/bash
set -e
python3 generate_synthetic_behavioral_features.py
python3 generate_synthetic_wearable.py
python3 train_attention_model.py
python3 rl_adaptive_demo.py
echo 'Demo complete. Artifacts: behavioral_features.csv, wearable_timeseries.csv, attention_model.joblib, rl_agent_summary.csv'
