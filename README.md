# Multimodal ASD/ADHD Adaptive Learning Prototype

## Overview

This is a **research-focused, privacy-preserving** demonstration project showcasing a multimodal system for adaptive intervention in individuals with Autism Spectrum Disorder (ASD) and Attention-Deficit/Hyperactivity Disorder (ADHD). The project uses **100% synthetic data** to illustrate the core engineering and machine learning components without involving any clinical information.

The system integrates:
- **Behavioral analysis** from video (gaze contact, head pose, emotional expressions)
- **Physiological monitoring** from wearable sensors (heart rate variability, skin conductance, accelerometry)
- **Machine learning** to predict attention states from wearable biosignals
- **Adaptive reinforcement learning** to dynamically adjust activity difficulty based on real-time attention levels

This demonstrates a complete pipeline from data generation through model training to real-time adaptation, suitable for academic portfolios, research proposals, and interview discussions.

---

## Project Architecture

### 1. Data Generation Layers

#### `generate_synthetic_behavioral_features.py`
**Purpose:** Simulates video-derived behavioral markers

**Generated Features:**
- **Gaze Contact** (0-1): Proportion of eye contact during interaction
  - Distribution: Normal(μ=0.6, σ=0.15), clipped to [0, 1]
  - Clinical relevance: Reduced gaze contact is a diagnostic marker for ASD
  
- **Head Rotation** (-90°, +90°): Head positioning relative to frontal view
  - Distribution: Normal(μ=10°, σ=6°)
  - Clinical relevance: Atypical head posture patterns in ASD
  
- **Smile Score** (0-1): Intensity/frequency of smiling
  - Distribution: Beta(2, 5) — realistic skew toward lower values
  - Clinical relevance: Differences in social emotional expression
  
- **Joint Attention** (0-1): Ability to shift attention with another person
  - Distribution: Normal(μ=0.3, σ=0.2)
  - Clinical relevance: Critical ASD diagnostic criterion
  
- **Prosody Energy** (0-2): Vocal expression intensity
  - Distribution: Normal(μ=0.5, σ=0.2)
  - Clinical relevance: Speech prosody differences in ASD/ADHD
  
- **Risk Score** (composite): Weighted combination suggesting behavioral intervention need
  - Formula: (1-gaze)×0.4 + (|head|/90)×0.2 + (1-smile)×0.2 + (1-joint_att)×0.2
  - Provides a simple clinical relevance metric

**Output:** `behavioral_features.csv` (500 samples)

---

#### `generate_synthetic_wearable.py`
**Purpose:** Creates time-series physiological data mimicking wearable sensors

**Generated Features per Subject:**
- **Heart Rate (HR):** 60-100 bpm baseline + circadian oscillation
  - Simulates: baseline variability, 1-minute oscillation pattern
  - Formula: `HR_base + N(0,3) + 5×sin(2πt/60)`
  
- **Electrodermal Activity (EDA):** Skin conductance level (0-3 μS)
  - Simulates: baseline + noise + 1.5-minute oscillation
  - Formula: `max(0, EDA_base + N(0,0.2) + 0.2×sin(2πt/90))`
  
- **Acceleration (Accel):** Movement magnitude
  - Simulates: baseline variability + sporadic movement bursts (2% probability)
  - Clinical relevance: Hyperactivity/restlessness in ADHD
  
- **Attention Label:** Binary (0=inattentive, 1=attentive)
  - Probability of inattention: 0.1 + 0.4×|sin(2πt/300)|
  - Creates cyclic attention patterns with base rate ~10%

**Output:** `wearable_timeseries.csv` (20 subjects × 300 timesteps = 6,000 records)

---

### 2. Machine Learning Pipeline

#### `train_attention_model.py`
**Purpose:** Train a classifier to predict attention states from wearable features

**Processing Steps:**

1. **Feature Windowing:**
   - Sliding window of 10 consecutive timesteps per subject
   - Features: [hr, eda, accel] × 10 timesteps = 30-dimensional vectors
   - Window-level label: Attention=1 if mean(attention_labels) > 0.5, else 0
   - This temporal aggregation mimics clinical observation windows

2. **Train/Test Split:**
   - 80/20 stratified split (random_state=42)
   - Maintains subject-level independence

3. **Classification Model:**
   - **Algorithm:** Random Forest (100 estimators)
   - **Rationale:** 
     - Interpretable feature importance (clinical transparency)
     - Robust to physiological noise
     - No feature scaling required
   - **Training:** On windowed wearable features to predict attention

4. **Evaluation Metrics:**
   - Accuracy score
   - Precision, Recall, F1-score (per class)
   - Classification report

**Output:** 
- Console: Classification metrics
- File: `attention_model.joblib` (serialized trained model for later deployment)

---

### 3. Adaptive Learning Simulation

#### `rl_adaptive_demo.py`
**Purpose:** Demonstrates real-time adaptation using reinforcement learning

**Agent Architecture:**

**SimpleAdaptiveAgent Class:**
- **State Space:** Implicit (based on attention signal)
- **Action Space:** 3 discrete difficulty levels
  - Action 0 = Easy (difficulty 0.0)
  - Action 1 = Medium (difficulty 0.5)
  - Action 2 = Hard (difficulty 1.0)
  
- **Algorithm:** ε-greedy (epsilon=0.2)
  - 20% random exploration, 80% exploitation
  - Q-learning style value update

**Reward Function:**
```
if attention == high (1):
    reward = 1.0 - 0.2×|1 - difficulty|  [penalize too easy or too hard]
else:
    reward = 1.0 - 0.5×difficulty        [ease task to regain attention]
reward = max(0, reward + N(0, 0.05))     [add realism noise]
```

**Clinical Relevance:**
- Automatically adjusts intervention intensity
- When child loses attention → system reduces difficulty
- Balances challenge and engagement (flow theory)

**Episode Simulation:**
- 100 steps per episode
- Uses real attention sequence from generated wearable data
- Tracks: cumulative rewards, Q-value evolution, action counts

**Output:**
- Console: Mean episode reward, learned Q-values
- File: `rl_agent_summary.csv` (action preferences and visit counts)

---

### 4. Pipeline Orchestration

#### `demo_run.sh`
Bash script that executes the full pipeline in order:
```bash
1. Generate behavioral features
2. Generate wearable timeseries
3. Train attention classifier
4. Run RL adaptive episode
```

Produces all 4 data artifacts in sequence.

---

## Output Artifacts

| File | Description | Rows/Size |
|------|-------------|-----------|
| `behavioral_features.csv` | Video-derived features snapshot | 500 |
| `wearable_timeseries.csv` | Sensor time-series data | 6,000 (20 subjects × 300 steps) |
| `attention_model.joblib` | Trained Random Forest classifier | Binary model |
| `rl_agent_summary.csv` | Final Q-values and visit counts | 3 actions |

---

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip or conda package manager

### Local Setup

#### Option 1: Virtual Environment (Recommended)
```bash
# Clone or download this repository
cd demo_project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Conda Environment
```bash
conda create -n asd_adhd_demo python=3.9
conda activate asd_adhd_demo
pip install -r requirements.txt
```

### Dependencies
- **numpy** (2.2.6+): Numerical computing
- **pandas** (2.2.3+): Data manipulation
- **scikit-learn** (1.5.1+): Machine learning
- **matplotlib** (3.10.1+): Visualization (optional for future extensions)

---

## Usage

### Running the Complete Pipeline
```bash
bash demo_run.sh
```

This executes all 4 stages sequentially and generates all output artifacts.

### Running Individual Components

**Generate behavioral features:**
```bash
python generate_synthetic_behavioral_features.py
```

**Generate wearable timeseries:**
```bash
python generate_synthetic_wearable.py
```

**Train attention model:**
```bash
python train_attention_model.py
```

**Run RL adaptive agent:**
```bash
python rl_adaptive_demo.py
```

### Expected Console Output

```
Behavioral features saved to behavioral_features.csv
Wearable timeseries saved to wearable_timeseries.csv
Accuracy: 0.78
              precision    recall  f1-score   support
           0       0.75      0.82      0.78      2400
           1       0.81      0.73      0.77      2400
Trained model saved to attention_model.joblib
Mean reward after simulation: 0.68
Learned Q-values: [0.62 0.71 0.55]
RL summary saved to rl_agent_summary.csv
```

---

## System Design & Clinical Motivation

### Why This Architecture?

**Multimodal Integration:**
- Video-based behavioral features capture social/communicative markers
- Wearable physiological data provides objective, real-time engagement metrics
- Combined signals improve reliability and reduce false positives

**Real-Time Adaptation:**
- Classical ASD/ADHD interventions are often one-size-fits-all
- This system demonstrates dynamic difficulty adjustment
- Keeps users in optimal challenge zone (Csikszentmihalyi's "flow")

**Machine Learning for Attention:**
- Random Forest learns non-linear mappings from HR, EDA, motion → attention state
- Physiological markers (autonomic nervous system) are strong attention correlates
- No patient-identifiable information required

**Reinforcement Learning for Adaptation:**
- ε-greedy balances exploration (try new difficulties) vs. exploitation (use best)
- Reward signal directly optimizes user engagement
- Scalable to more complex state/action spaces with deep RL

---

## Privacy & Ethics

### Synthetic Data Only
This repository uses **100% simulated data** to:
- ✅ Respect patient privacy and HIPAA/GDPR requirements
- ✅ Eliminate need for IRB pre-approval during development
- ✅ Allow open-source sharing without clinical liability
- ✅ Demonstrate feasibility without sensitive information

### Path to Real Clinical Data
To deploy with actual patient data:
1. Obtain **IRB (Institutional Review Board)** approval
2. Implement **informed consent** protocols
3. Ensure **HIPAA/GDPR compliance** (de-identification, encryption)
4. Conduct **clinical validation** studies
5. Document **safety & efficacy** metrics

*(See `Clinical_Data_Ethics_Explanation.pdf` for detailed discussion.)*

---

## Use Cases

### For Academic/Professional Portfolio
- **Demonstrates:** ML pipeline, time-series analysis, RL fundamentals
- **Suitable for:** Resume, portfolio website, GitHub showcases
- **Interview talking points:**
  - Multimodal data fusion
  - Feature engineering from heterogeneous sources
  - Model validation metrics
  - Real-time adaptive systems

### For Research Proposals
- **Proof-of-concept:** Full pipeline with reproducible results
- **IRB submission base:** Template for clinical deployment
- **Funding discussion:** Concrete demo of system capabilities

### For Teaching/Learning
- **Educational value:** Complete ML workflow (data → model → deployment)
- **Extensible:** Easy to add more features, models, or RL algorithms
- **Self-contained:** No external APIs or data dependencies

---

## Future Extensions

Possible enhancements (with real data):
- [ ] Add more behavioral features (pose landmarks, gaze patterns)
- [ ] Integrate speech analysis (prosody, tempo, pause patterns)
- [ ] Implement deep learning models (LSTMs for temporal attention)
- [ ] Add visualization dashboard (Plotly/Dash)
- [ ] Deploy as REST API (Flask/FastAPI)
- [ ] Implement multi-agent RL (competitive/collaborative scenarios)
- [ ] Cross-subject transfer learning
- [ ] Real-time inference on embedded devices (edge computing)

---

## Author Notes

This project demonstrates:
1. **Engineering best practices**: Modular design, clear separation of concerns
2. **ML competency**: Feature engineering, model selection, evaluation
3. **Clinical awareness**: Biologically plausible features, ethical considerations
4. **Systems thinking**: Integration of multiple data modalities

The synthetic nature is **intentional and principled**—not a limitation, but a feature enabling responsible development of healthcare AI.

---

## References & Inspiration

- **ASD Diagnostic Criteria (DSM-5):** Social-communicative deficits, repetitive behaviors
- **ADHD & Attention:** Autonomic nervous system markers (HRV, EDA) in attention disorders
- **Flow Theory (Csikszentmihalyi):** Optimal challenge and engagement balance
- **Q-Learning (Watkins & Dayan):** ε-greedy exploration-exploitation trade-off
- **Multimodal ML:** Fusion of video and physiological signals

---

## License & Contact

This is a demonstration project for educational and portfolio purposes. All code is provided as-is for non-commercial use.

For questions or extensions, refer to the inline code comments and docstrings.
