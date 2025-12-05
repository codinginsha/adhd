@echo off
REM Windows batch script to run the full demo pipeline

echo Running Multimodal ASD/ADHD Demo Pipeline...
echo.

echo Step 1: Generating behavioral features...
python generate_synthetic_behavioral_features.py
if errorlevel 1 (
    echo Error generating behavioral features
    exit /b 1
)
echo.

echo Step 2: Generating wearable timeseries...
python generate_synthetic_wearable.py
if errorlevel 1 (
    echo Error generating wearable timeseries
    exit /b 1
)
echo.

echo Step 3: Training attention model...
python train_attention_model.py
if errorlevel 1 (
    echo Error training attention model
    exit /b 1
)
echo.

echo Step 4: Running RL adaptive demo...
python rl_adaptive_demo.py
if errorlevel 1 (
    echo Error running RL demo
    exit /b 1
)
echo.

echo.
echo ===============================================
echo Demo complete! Artifacts generated:
echo   - behavioral_features.csv
echo   - wearable_timeseries.csv
echo   - attention_model.joblib
echo   - rl_agent_summary.csv
echo ===============================================
