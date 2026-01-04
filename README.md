# Real-Time Dump Truck Anomaly Detection System

A production-grade industrial monitoring platform that leverages deep learning autoencoders for real-time anomaly detection in mining dump truck components. This system provides predictive maintenance capabilities through unsupervised machine learning, component-wise health monitoring, and explainable AI-driven root cause analysis.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Key Features](#key-features)
4. [System Architecture](#system-architecture)
5. [Technology Stack](#technology-stack)
6. [Datasets](#datasets)
7. [Autoencoder Training Pipeline](#autoencoder-training-pipeline)
8. [Anomaly Detection Logic](#anomaly-detection-logic)
9. [Explainable AI Implementation](#explainable-ai-implementation)
10. [Natural Language Explanations](#natural-language-explanations)
11. [Frontend Dashboards](#frontend-dashboards)
12. [Failure Injection System](#failure-injection-system)
13. [Performance and Design](#performance-and-design)
14. [Installation and Setup](#installation-and-setup)
15. [Deployment](#deployment)
16. [Project Use Cases](#project-use-cases)
17. [Future Improvements](#future-improvements)
18. [License](#license)

---

## Project Overview

The Real-Time Dump Truck Anomaly Detection System is an end-to-end industrial monitoring solution designed for heavy machinery operating in mining and construction environments. The system continuously analyzes sensor data from four critical subsystems of a dump truck:

- **Engine**: Temperature, oil pressure, and fuel consumption monitoring
- **Hydraulic System**: Pressure levels and load capacity tracking
- **Wheel Assembly**: Vibration patterns, brake temperatures, and rotational speed
- **Chassis**: Structural vibration and payload distribution

The platform employs autoencoder neural networks trained exclusively on normal operational data to identify deviations from expected behavior patterns. When sensor readings deviate significantly from the learned normal distribution, the system flags potential anomalies and provides human-readable explanations of the probable root cause.

### Why This Project Exists

Heavy mining equipment represents substantial capital investment, with individual dump trucks costing several million dollars. Unplanned downtime due to component failure results in:

- Direct repair costs and replacement parts
- Lost productivity during repair periods
- Cascading delays in mining operations
- Safety hazards for operators and site personnel
- Environmental risks from fluid leaks and mechanical failures

Traditional maintenance approaches (reactive or scheduled) are either too late or too frequent. This system enables **condition-based predictive maintenance**, allowing operators to:

- Detect degradation before catastrophic failure
- Schedule maintenance during planned downtime
- Optimize spare parts inventory
- Extend component service life
- Reduce total cost of ownership

---

## Problem Statement

Mining dump trucks operate in extreme conditions: high ambient temperatures, dust exposure, heavy loads, and continuous operation cycles. These conditions accelerate wear on mechanical, hydraulic, and electrical systems. The challenge is threefold:

1. **Data Complexity**: Dozens of sensors generate continuous streams of multivariate time-series data with complex interdependencies
2. **Labeled Data Scarcity**: True failure events are rare (and expensive), making supervised learning impractical for most industrial applications
3. **Interpretability Requirements**: Operators need actionable insights, not black-box predictions

This system addresses all three challenges through:

- Component-specific autoencoder models that learn normal behavior patterns
- Unsupervised learning that requires only normal operational data for training
- Feature-wise reconstruction error analysis that identifies which sensors contributed to an anomaly

---

## Key Features

### Real-Time Sensor Simulation
- Continuous generation of realistic sensor data based on statistical distributions derived from historical operational data
- Time-synchronized data streams across all four components
- Configurable noise levels and operational parameters

### Component-Wise Anomaly Detection
- Independent autoencoder models for each subsystem (Engine, Hydraulic, Tyres, Chassis)
- Parallel anomaly evaluation with component-specific thresholds
- Global anomaly aggregation for system-wide health assessment

### Autoencoder-Based Unsupervised Learning
- Deep neural network architecture with encoder-decoder structure
- Training exclusively on normal operational data
- Reconstruction error as anomaly score

### Explainable AI (Root Cause Analysis)
- Feature-wise reconstruction error attribution
- Percentage contribution calculation per sensor
- Natural language explanation generation

### Interactive Dashboards
- Main overview dashboard with all system metrics
- Component-specific detailed views
- Real-time time-series visualizations using Plotly.js

### Industrial Visualization
- SVG-based dump truck schematic
- Red glow effect on anomalous components
- Animated grey smoke visualization during failures
- Component highlighting and status indicators

### Failure Injection for Testing
- Manual anomaly injection via UI button
- Realistic multi-component failure scenarios
- Weighted probability distribution for failure selection
- Immediate visual feedback (less than 100ms response)

### High-Performance Architecture
- Sub-100ms response times for all user interactions
- Non-blocking backend operations
- Efficient model inference with pre-loaded weights
- Thread-safe shared state management

---

## System Architecture

The application follows a modular architecture with clear separation between data generation, anomaly detection, state management, and presentation layers.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Main      │  │   Engine    │  │  Hydraulic  │  │   Tyres     │        │
│  │ Dashboard   │  │  Dashboard  │  │  Dashboard  │  │  Dashboard  │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                   │                                          │
│                          AJAX Polling (500ms)                               │
│                                   │                                          │
└───────────────────────────────────┼──────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼──────────────────────────────────────────┐
│                              API LAYER                                       │
│                                   │                                          │
│  ┌────────────────────────────────▼────────────────────────────────────────┐│
│  │                      /api/live_state                                    ││
│  │              Unified endpoint for all dashboards                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                   │                                          │
│  ┌─────────────┐  ┌─────────────┐ │ ┌─────────────┐  ┌─────────────┐        │
│  │ /simulate   │  │ /toggle     │ │ │ /api/       │  │ /api/state  │        │
│  │   _data     │  │ _injection  │ │ │ component/  │  │  (debug)    │        │
│  └─────────────┘  └─────────────┘ │ └─────────────┘  └─────────────┘        │
│                                   │                                          │
└───────────────────────────────────┼──────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼──────────────────────────────────────────┐
│                         SHARED STATE LAYER                                   │
│                                   │                                          │
│  ┌────────────────────────────────▼────────────────────────────────────────┐│
│  │                         SharedState Class                               ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  ││
│  │  │  Components  │  │  Time Series │  │  Injection   │                  ││
│  │  │    State     │  │    Buffers   │  │    State     │                  ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  ││
│  │                    Thread-Safe Lock                                     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼──────────────────────────────────────────┐
│                         ML INFERENCE LAYER                                   │
│                                   │                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      Data Generation & Preprocessing                    ││
│  │  • Statistical sensor simulation based on historical distributions      ││
│  │  • StandardScaler normalization per component                           ││
│  │  • Failure injection multipliers when enabled                           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                   │                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Engine    │  │  Hydraulic  │  │   Tyres     │  │   Chassis   │        │
│  │ Autoencoder │  │ Autoencoder │  │ Autoencoder │  │ Autoencoder │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                │                │                │                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Scaler    │  │   Scaler    │  │   Scaler    │  │   Scaler    │        │
│  │   + PCA     │  │   + PCA     │  │   + PCA     │  │   + PCA     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Architectural Principles

#### Single Source of Truth
All dashboards read from a centralized `SharedState` object. No dashboard generates its own data independently, ensuring consistency across all views.

#### Separation of Control Plane and Data Plane
- **Control Plane**: Handles user interactions (failure injection toggles, navigation)
- **Data Plane**: Manages sensor data flow, anomaly detection, and state updates

#### Non-Blocking Operations
- Model inference is performed synchronously but efficiently (pre-loaded models)
- Time series buffers use fixed-size rolling windows (250 points maximum)
- Thread locks protect shared state during concurrent access

#### Instant Page Rendering
- HTML templates render immediately without data blocking
- Client-side JavaScript fetches data asynchronously via AJAX
- Navigation between dashboards does not trigger model reloading

---

## Technology Stack

### Backend Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Core programming language |
| **Flask** | 3.0.0+ | Lightweight WSGI web framework |
| **NumPy** | 1.24.0+ | Numerical computations and array operations |
| **Pandas** | 2.0.0+ | Dataset manipulation and preprocessing |
| **Scikit-learn** | 1.3.0+ | StandardScaler and PCA transformations |
| **TensorFlow** | 2.15.0+ | Deep learning framework |
| **Keras** | (TensorFlow) | High-level neural network API |
| **Joblib** | 1.3.0+ | Model and artifact serialization |

### Frontend Technologies

| Technology | Purpose |
|------------|---------|
| **HTML5** | Semantic page structure |
| **CSS3** | Styling, animations, and visual effects |
| **JavaScript (ES6+)** | Client-side logic and API communication |
| **Plotly.js** | Interactive time-series charts and 3D visualizations |
| **SVG** | Vector graphics for industrial truck visualization |

### Machine Learning Stack

| Component | Implementation |
|-----------|----------------|
| **Model Architecture** | Dense autoencoder with batch normalization and dropout |
| **Loss Function** | Mean Squared Error (MSE) |
| **Optimizer** | Adam |
| **Feature Scaling** | StandardScaler (zero mean, unit variance) |
| **Dimensionality Reduction** | PCA (for visualization) |
| **Anomaly Scoring** | Reconstruction error thresholding |

### Deployment Stack

| Technology | Purpose |
|------------|---------|
| **Gunicorn** | Production WSGI HTTP server |
| **Render** | Cloud deployment platform |
| **Procfile** | Process type declaration for Render |
| **requirements.txt** | Python dependency specification |

---

## Datasets

The system uses four independent CSV datasets, one per component. This separation is intentional and critical for accurate anomaly detection.

### Dataset Structure

Each dataset follows a consistent schema:

```
timestamp,sensor_1,sensor_2,...,sensor_n,label
```

Where:
- `timestamp`: ISO 8601 formatted datetime (hourly intervals)
- `sensor_*`: Component-specific sensor readings
- `label`: Binary indicator (0 = normal, 1 = anomaly)

### Component Datasets

#### Engine Dataset (`datasets/engine.csv`)

| Column | Description | Unit | Normal Range |
|--------|-------------|------|--------------|
| `timestamp` | Observation time | datetime | - |
| `engine_temp_C` | Engine block temperature | Celsius | 75-95 |
| `oil_pressure_bar` | Lubrication system pressure | Bar | 3.5-5.5 |
| `fuel_rate_lph` | Fuel consumption rate | Liters/hour | 35-55 |
| `label` | Anomaly indicator | binary | 0/1 |

#### Hydraulic Dataset (`datasets/hydraulic.csv`)

| Column | Description | Unit | Normal Range |
|--------|-------------|------|--------------|
| `timestamp` | Observation time | datetime | - |
| `hydraulic_pressure_bar` | Main hydraulic circuit pressure | Bar | 190-235 |
| `load_tons` | Current payload weight | Tons | 45-65 |
| `label` | Anomaly indicator | binary | 0/1 |

#### Tyres Dataset (`datasets/wheels.csv`)

| Column | Description | Unit | Normal Range |
|--------|-------------|------|--------------|
| `timestamp` | Observation time | datetime | - |
| `vibration_mm_s` | Wheel assembly vibration | mm/s | 1.5-3.5 |
| `brake_temp_C` | Brake disc temperature | Celsius | 105-135 |
| `speed_kmph` | Ground speed | km/h | 25-50 |
| `label` | Anomaly indicator | binary | 0/1 |

#### Chassis Dataset (`datasets/chassis.csv`)

| Column | Description | Unit | Normal Range |
|--------|-------------|------|--------------|
| `timestamp` | Observation time | datetime | - |
| `vibration_mm_s` | Structural vibration | mm/s | 1.5-3.5 |
| `load_tons` | Distributed payload | Tons | 45-65 |
| `label` | Anomaly indicator | binary | 0/1 |

### Why Datasets Are Not Merged

Merging datasets would violate fundamental assumptions of the anomaly detection approach:

1. **Feature Space Contamination**: Each component has its own normal operating envelope. Merging would create spurious correlations.

2. **Threshold Calibration**: Anomaly thresholds are calculated per-component based on reconstruction error distributions. Merged data would require a single global threshold, reducing sensitivity.

3. **Root Cause Isolation**: Separate models enable precise identification of which component is failing.

4. **Computational Efficiency**: Smaller models (2-3 features) train faster and infer more quickly than a single large model.

### Time Alignment

All datasets are time-synchronized at hourly intervals starting from `2025-01-01 00:00:00`. This alignment ensures:

- Consistent temporal correlation analysis
- Valid multi-component failure scenario generation
- Accurate historical replay capabilities

---

## Autoencoder Training Pipeline

### Why Autoencoders

Autoencoders are neural networks trained to reconstruct their input. When trained exclusively on normal data:

1. The model learns a compressed representation (encoding) of normal patterns
2. Anomalous inputs cannot be reconstructed accurately
3. High reconstruction error signals deviation from normal behavior

This approach is ideal for industrial applications because:

- No labeled anomaly data required for training
- Captures complex, non-linear relationships between sensors
- Provides interpretable anomaly scores (reconstruction error)

### Training Process

The training pipeline (`train_component_models.py`) follows these steps for each component:

#### Step 1: Data Loading and Filtering

```python
df = pd.read_csv(f'datasets/{component_name}.csv')
df_normal = df[df['label'] == 0]  # CRITICAL: Only normal data
```

#### Step 2: Feature Extraction

```python
sensors = ['engine_temp_C', 'oil_pressure_bar', 'fuel_rate_lph']
X_normal = df_normal[sensors].values
```

#### Step 3: Train-Test Split

```python
n_train = int(len(X_normal) * 0.8)
X_train = X_normal[:n_train]
X_test = X_normal[n_train:]
```

#### Step 4: Feature Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Scaling ensures:
- Zero mean: Features centered around 0
- Unit variance: Features have comparable magnitudes
- Stable gradient descent during training

#### Step 5: Autoencoder Architecture

```python
def build_autoencoder(input_dim, encoding_dim):
    encoder_input = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_dim, activation='relu')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(hidden_dim // 2, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    encoded = layers.Dense(encoding_dim, activation='relu')(x)
    
    x = layers.Dense(hidden_dim // 2, activation='relu')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='linear')(x)
    
    autoencoder = keras.Model(encoder_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder
```

Architecture features:
- **Encoder**: Compresses input to lower-dimensional latent space
- **Decoder**: Reconstructs original input from latent representation
- **Batch Normalization**: Stabilizes training, reduces internal covariate shift
- **Dropout (0.2)**: Prevents overfitting, improves generalization
- **ReLU Activation**: Non-linearity for hidden layers
- **Linear Output**: Continuous reconstruction values

#### Step 6: Model Training

```python
history = autoencoder.fit(
    X_train_scaled, X_train_scaled,  # Input = Target (reconstruction)
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)
```

#### Step 7: Threshold Calculation

```python
X_test_pred = autoencoder.predict(X_test_scaled, verbose=0)
test_errors = np.mean(np.square(X_test_scaled - X_test_pred), axis=1)
threshold = np.percentile(test_errors, 99)
```

The 99th percentile threshold ensures:
- 99% of normal samples fall below threshold
- Only significant deviations trigger anomaly flags
- Balance between sensitivity and false positive rate

#### Step 8: Artifact Serialization

```python
autoencoder.save(f'models/{component_name}_autoencoder.h5')
joblib.dump(scaler, f'models/{component_name}_scaler.pkl')
joblib.dump(pca, f'models/{component_name}_pca.pkl')
joblib.dump(threshold, f'models/{component_name}_threshold.pkl')
```

### Model Artifacts

Each component produces five artifacts:

| Artifact | Format | Purpose |
|----------|--------|---------|
| `{component}_autoencoder.h5` | HDF5 | Trained neural network weights |
| `{component}_scaler.pkl` | Pickle | StandardScaler parameters |
| `{component}_pca.pkl` | Pickle | PCA transformation matrix |
| `{component}_threshold.pkl` | Pickle | Anomaly threshold value |
| `{component}_stats.pkl` | Pickle | Training statistics |

---

## Anomaly Detection Logic

### Runtime Detection Pipeline

When new sensor data arrives, the following process executes:

#### Step 1: Data Preprocessing

```python
X_input = raw_data.reshape(1, -1)
X_scaled = model_info['scaler'].transform(X_input)
```

#### Step 2: Model Reconstruction

```python
X_pred = model_info['model'].predict(X_scaled, verbose=0)
```

#### Step 3: Reconstruction Error Computation

```python
comp_diff = X_scaled - X_pred
comp_recon_error = float(np.mean(np.square(comp_diff)))
```

The reconstruction error is the Mean Squared Error (MSE) between input and reconstruction:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2$$

Where:
- $x_i$ = scaled input feature value
- $\hat{x}_i$ = reconstructed feature value
- $n$ = number of features

#### Step 4: Threshold Comparison

```python
comp_anomaly = comp_recon_error > comp_threshold
```

#### Step 5: Component-Level Aggregation

```python
affected_components = {
    'engine': False,
    'hydraulic': False,
    'chassis': False,
    'tyres': False
}

for comp_name in components:
    # ... detection logic ...
    affected_components[comp_name] = comp_anomaly
```

### Anomaly Severity Concept

While the current implementation uses binary anomaly flags, the reconstruction error magnitude provides implicit severity information:

| Error Ratio | Severity Level |
|-------------|----------------|
| 1.0 - 1.5x threshold | Minor deviation |
| 1.5 - 2.5x threshold | Moderate anomaly |
| 2.5 - 4.0x threshold | Significant anomaly |
| > 4.0x threshold | Critical anomaly |

---

## Explainable AI Implementation

### Why Not SHAP or LIME

Traditional explainability methods like SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) are designed for classification or regression models. They have significant limitations for autoencoder-based anomaly detection:

| Method | Limitation for Autoencoders |
|--------|----------------------------|
| **SHAP** | Requires a scalar output; autoencoders output vectors |
| **SHAP** | Computationally expensive for real-time applications |
| **LIME** | Assumes local linearity; autoencoders may have non-linear reconstruction |
| **LIME** | Perturbation-based; unstable for multivariate sensor data |
| **Both** | Designed for model predictions, not reconstruction errors |

### Feature-Wise Reconstruction Error Attribution

This system implements a direct, deterministic approach:

**Principle**: The feature(s) that the autoencoder struggles to reconstruct are the feature(s) causing the anomaly.

#### Mathematical Formulation

For input vector $\mathbf{x} = [x_1, x_2, ..., x_n]$ and reconstruction $\hat{\mathbf{x}} = [\hat{x}_1, \hat{x}_2, ..., \hat{x}_n]$:

**Per-feature squared error:**

$$e_i = (x_i - \hat{x}_i)^2$$

**Total reconstruction error:**

$$E_{total} = \sum_{i=1}^{n} e_i$$

**Feature contribution percentage:**

$$C_i = \frac{e_i}{E_{total}} \times 100\%$$

#### Implementation

```python
def calculate_feature_contributions(X_scaled, X_pred, feature_names):
    # Per-feature squared error
    feature_errors = np.square(X_scaled.flatten() - X_pred.flatten())
    total_error = np.sum(feature_errors)
    
    contributions = []
    for i, feature_name in enumerate(feature_names):
        contribution_pct = (feature_errors[i] / total_error) * 100
        contributions.append({
            'feature': feature_name,
            'contribution': round(float(contribution_pct), 1),
            'error': round(float(feature_errors[i]), 6)
        })
    
    # Sort by contribution descending
    contributions.sort(key=lambda x: x['contribution'], reverse=True)
    return contributions
```

### Advantages of This Approach

1. **Deterministic**: Same input always produces same explanation
2. **Real-time**: O(n) computation, no iterative sampling
3. **Interpretable**: Direct mapping from sensor to contribution
4. **Proportional**: Percentages sum to 100%, intuitive for operators
5. **Consistent**: No randomness or perturbation artifacts

---

## Natural Language Explanations

### Conversion from Numeric to Text

The system transforms numeric feature contributions into operator-friendly natural language:

#### Severity Adjectives

```python
def get_severity_adjective(contribution):
    if contribution >= 60:
        return "critically"
    elif contribution >= 40:
        return "significantly"
    elif contribution >= 25:
        return "notably"
    elif contribution >= 15:
        return "moderately"
    else:
        return "slightly"
```

#### Component-Specific Cause Templates

```python
COMPONENT_CAUSE_TEMPLATES = {
    'engine': {
        'engine_temperature': ['overheating condition', 'thermal stress', ...],
        'oil_pressure': ['lubrication system stress', 'potential oil pump degradation', ...],
        'fuel_rate': ['fuel injection irregularity', 'combustion inefficiency', ...]
    },
    # ... other components
}
```

### Example Explanations

#### Engine Overheating

**Numeric data:**
- engine_temperature: 67.3% contribution
- oil_pressure: 22.1% contribution
- fuel_rate: 10.6% contribution

**Generated explanation:**
> "Engine anomaly detected primarily due to critically elevated engine temperature, contributing 67% of the total deviation. This indicates overheating condition. Elevated oil pressure (22%) further indicates lubrication system stress."

#### Hydraulic Pressure Failure

**Numeric data:**
- hydraulic_pressure: 78.5% contribution
- fluid_temp: 21.5% contribution

**Generated explanation:**
> "Hydraulic system anomaly detected primarily due to critically elevated hydraulic pressure, contributing 79% of the total deviation. This indicates possible blockage or valve malfunction."

#### Wheel Vibration Anomaly

**Numeric data:**
- wheel_vibration: 52.4% contribution
- brake_temperature: 31.2% contribution
- wheel_speed: 16.4% contribution

**Generated explanation:**
> "Wheel assembly anomaly detected primarily due to significantly elevated wheel vibration, contributing 52% of the total deviation. This indicates imbalance or surface irregularities. Elevated brake temperature (31%) further indicates brake system overheating."

#### Chassis Overload

**Numeric data:**
- load_weight: 71.8% contribution
- chassis_vibration: 28.2% contribution

**Generated explanation:**
> "Chassis anomaly detected primarily due to critically elevated load weight, contributing 72% of the total deviation. This indicates excessive payload. Elevated structural vibration (28%) further indicates structural resonance."

---

## Frontend Dashboards

### Main Overview Dashboard

The primary interface provides a holistic view of truck health:

**Components:**
- System status indicator (NORMAL/ANOMALY)
- Real-time sensor readings for all 8 sensors
- 3D PCA visualization of system state
- Time-series charts (Engine Temperature, Vibration)
- Interactive truck schematic with component highlighting
- Control panel for failure injection

**Visual Features:**
- Green glow for normal components
- Red pulsing glow for anomalous components
- Animated grey smoke effect for active failures
- Horizontal scrolling time-series graphs

### Component-Specific Dashboards

Each component has a dedicated interface accessible via:
- `/component/engine` - Engine monitoring
- `/component/hydraulic` - Hydraulic system monitoring
- `/component/tyres` - Tyre assembly monitoring
- `/component/chassis` - Chassis monitoring

**Features:**
- Component-specific sensor readings
- Reconstruction error vs. threshold visualization
- Root cause analysis panel with:
  - Feature contribution bar chart
  - Natural language explanation
- Component time-series data

### Real-Time Data Flow

```
┌──────────────┐     500ms interval     ┌──────────────┐
│   Browser    │ ─────────────────────> │  /api/live   │
│  JavaScript  │ <───────────────────── │   _state     │
└──────────────┘     JSON response      └──────────────┘
       │
       ▼
┌──────────────┐
│  Plotly.js   │
│   Update     │
└──────────────┘
```

### Synchronization Across Views

All dashboards poll the same `/api/live_state` endpoint, which:

1. Triggers fresh data generation via `simulate_data()`
2. Updates the SharedState object
3. Returns component-specific data slices

This ensures all views show consistent data, regardless of navigation timing.

---

## Failure Injection System

### Purpose

The failure injection system enables:

1. **Testing**: Verify anomaly detection sensitivity
2. **Training**: Familiarize operators with anomaly indicators
3. **Demonstration**: Showcase system capabilities to stakeholders
4. **Validation**: Confirm explanation generation accuracy

### Injection Mechanism

#### Activation

```python
@app.route('/toggle_injection', methods=['POST'])
def toggle_injection():
    with shared_state.lock:
        shared_state.inject_anomaly = not shared_state.inject_anomaly
        if current_state:
            shared_state.active_failures = select_failure_scenario()
```

#### Failure Scenario Selection

Scenarios are selected based on realistic probability weights:

```python
FAILURE_SCENARIOS = [
    (['engine'], 0.30),                  # 30% - Engine-only failure
    (['hydraulic'], 0.22),               # 22% - Hydraulic-only failure
    (['tyres'], 0.13),                   # 13% - Tyres-only failure
    (['chassis'], 0.05),                 # 5%  - Chassis-only failure
    (['engine', 'hydraulic'], 0.12),     # 12% - Engine + Hydraulic
    (['tyres', 'chassis'], 0.10),        # 10% - Tyres + Chassis
    (['hydraulic', 'tyres'], 0.08)       # 8%  - Hydraulic + Tyres
]
```

#### Sensor Value Manipulation

```python
if inject and component in active_failures:
    for sensor_idx in COMPONENT_SENSORS[component]['indices']:
        spike_magnitude = 8 + np.random.rand() * 4  # 8-12x standard deviation
        raw_data[sensor_idx] = means[sensor_idx] + spike_magnitude * stds[sensor_idx]
```

### Response Time Guarantee

The system guarantees sub-100ms response to injection toggle:

- State change is immediate (in-memory)
- No database writes or file I/O
- Next data poll reflects new state
- Visual update on next render cycle

---

## Performance and Design

### Low-Latency Architecture

| Operation | Target Latency | Implementation |
|-----------|---------------|----------------|
| Page navigation | < 50ms | Static HTML, AJAX data loading |
| Injection toggle | < 100ms | In-memory state, no I/O |
| Data polling | < 200ms | Pre-loaded models, efficient inference |
| Chart update | < 50ms | Plotly.js optimized rendering |

### Efficiency Optimizations

1. **Model Pre-loading**: All models loaded at server startup
2. **Fixed-Size Buffers**: Time series limited to 250 points
3. **Thread-Safe Access**: Lock-protected shared state
4. **Efficient Serialization**: JSON for API responses
5. **Minimal DOM Updates**: Targeted Plotly chart updates

### Industrial-Grade Reliability

- **Graceful Degradation**: Missing models fallback to injection state
- **Error Handling**: Try-catch blocks around ML inference
- **State Isolation**: Thread locks prevent race conditions
- **Logging**: Periodic status output for debugging

---

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (optional, for cloning)

### Local Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd ANOMALY_DETECTOR
```

2. **Create virtual environment:**

```bash
python -m venv venv
```

3. **Activate virtual environment:**

**Windows:**
```bash
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

4. **Install dependencies:**

```bash
pip install -r requirements.txt
```

5. **Run the application:**

```bash
python app.py
```

6. **Access the dashboard:**

Open a browser and navigate to: `http://127.0.0.1:5000`

### Requirements File

```
flask>=3.0.0
gunicorn>=21.2.0
tensorflow>=2.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
Werkzeug>=3.0.0
```

---

## Deployment

### Render Deployment

This project is configured for deployment on Render.com.

#### Project Structure for Deployment

```
project-root/
├── app.py                      # Main Flask application (entry point)
├── requirements.txt            # Python dependencies
├── Procfile                    # Render start command
├── render.yaml                 # Render configuration (optional)
├── autoencoder_model.h5        # Main ML model
├── scaler.pkl                  # Data scaler
├── pca.pkl                     # PCA transformer
├── threshold.pkl               # Anomaly threshold
├── data_stats.pkl              # Data statistics
├── engine_autoencoder.h5       # Engine component model
├── engine_scaler.pkl           # ... component scalers
├── hydraulic_autoencoder.h5    # Hydraulic component model
├── wheels_autoencoder.h5       # Wheels component model
├── chassis_autoencoder.h5      # Chassis component model
├── templates/                  # HTML templates
│   ├── index.html
│   ├── engine.html
│   ├── hydraulic.html
│   ├── wheels.html
│   └── chassis.html
└── static/                     # Static files (CSS, JS, images)
```

#### Procfile Configuration

```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 --keep-alive 5 --log-level info
```

- **gunicorn**: Production-grade WSGI server
- **workers 1**: Single worker (TensorFlow memory considerations)
- **threads 4**: Multi-threading for concurrent real-time requests
- **timeout 120**: Extended timeout for model loading
- **keep-alive 5**: Connection keep-alive for real-time updates
- **log-level info**: Visible logs in Render dashboard

#### Render Start Command

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 --keep-alive 5 --log-level info
```

#### Deployment Steps

1. **Connect repository to Render:**
   - Create new Web Service
   - Connect GitHub/GitLab repository
   - Select branch to deploy

2. **Configure environment:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: (auto-detected from Procfile)

3. **Set environment variables:**
   - `PYTHON_VERSION`: 3.11.0 (recommended)
   - `TF_CPP_MIN_LOG_LEVEL`: 2 (suppress TensorFlow warnings)

4. **Deploy:**
   - Render automatically builds and deploys on push

#### Real-Time Execution Guarantees

- **Models loaded at startup**: All ML models are loaded once when the server starts, not on each request
- **In-memory state**: SharedState class maintains real-time data in memory for instant access
- **Thread-safe operations**: All data updates use thread locks for concurrent request handling
- **No file I/O during inference**: Model inference happens entirely in memory
- **API response time < 100ms**: Optimized for real-time dashboard updates

#### Alternative Deployment Platforms

The application can also be deployed on:

- **Heroku**: Same Procfile configuration
- **AWS Elastic Beanstalk**: Add `.ebextensions` configuration
- **Google Cloud Run**: Containerize with Dockerfile
- **Azure App Service**: Configure Python web app

---

## Project Use Cases

### Industrial Applications

1. **Predictive Maintenance**
   - Monitor equipment health continuously
   - Schedule maintenance based on condition
   - Reduce unplanned downtime

2. **Industrial IoT Integration**
   - Connect to SCADA systems
   - Integrate with historian databases
   - Feed into enterprise asset management

3. **Fleet Monitoring**
   - Scale to multiple trucks
   - Centralized dashboard for fleet managers
   - Comparative health analysis

### Educational Applications

4. **Academic Project**
   - Demonstrates ML pipeline end-to-end
   - Covers web development + machine learning
   - Suitable for capstone/final year projects

5. **Learning Platform**
   - Hands-on autoencoder implementation
   - Real-time system architecture
   - Explainable AI techniques

### Professional Development

6. **Portfolio Project**
   - Demonstrates full-stack capabilities
   - Shows ML engineering skills
   - Industry-relevant problem domain

7. **Interview Preparation**
   - Deep technical implementation details
   - Discussion points for system design
   - ML deployment best practices

---

## Future Improvements

### Enhanced Analytics

1. **Severity Scoring System**
   - Quantified severity levels (1-10 scale)
   - Alert prioritization based on severity
   - Trend analysis for degradation patterns

2. **Remaining Useful Life (RUL) Prediction**
   - Time-to-failure estimation
   - Maintenance window optimization
   - Component lifecycle tracking

### Scalability

3. **Fleet-Level Monitoring**
   - Multi-truck dashboard
   - Aggregated health metrics
   - Cross-fleet anomaly correlation

4. **Database Integration**
   - Historical data persistence
   - Long-term trend analysis
   - Audit trail for anomalies

### Advanced ML

5. **Model Drift Detection**
   - Automatic threshold recalibration
   - Distribution shift monitoring
   - Periodic retraining triggers

6. **Ensemble Methods**
   - Multiple model architectures
   - Voting-based anomaly decisions
   - Reduced false positive rates

### Operations

7. **Maintenance Recommendation Engine**
   - Automated work order generation
   - Parts inventory integration
   - Technician scheduling suggestions

8. **Mobile Application**
   - Responsive design optimization
   - Push notifications for anomalies
   - Offline capability for remote sites

---

## License

This project is developed for educational and demonstration purposes. Please refer to the repository for specific licensing terms.

---

## Acknowledgments

This system represents the application of modern machine learning techniques to real-world industrial challenges. The architecture and implementation draw from established practices in:

- Industrial IoT systems
- Predictive maintenance frameworks
- Explainable AI research
- Real-time web application design

---

## Contact

For questions, issues, or contributions, please open an issue in the repository or contact the project maintainers.

---

**Version**: 1.0.0  
**Last Updated**: January 2026
