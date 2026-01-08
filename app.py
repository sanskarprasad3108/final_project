"""
Dump Truck Multi-Interface Anomaly Detection System
====================================================
Real-time anomaly detection with SHARED STATE architecture.

ARCHITECTURE:
- ONE SharedState object holds ALL real-time data
- ALL routes READ from this shared state
- NO route generates its own data
- Component interfaces are synchronized views of the same data

PERFORMANCE GUARANTEES:
- Button click response < 100ms
- No blocking calls on navigation
- No model reloading on route change
- No duplicated data generation intervals
"""

import os
import sys
import logging

# Configure logging for Render visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, jsonify, request
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
import threading
import time
from datetime import datetime

# Base directory for resolving paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=BASE_DIR, static_folder=BASE_DIR)

# ============================================================
# SHARED REAL-TIME STATE - SINGLE SOURCE OF TRUTH
# ============================================================
class SharedState:
    """
    Centralized state manager for the entire system.
    ALL interfaces read from this state.
    NO interface generates data independently.
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        
        # Current timestamp
        self.timestamp = datetime.now().isoformat()
        self.tick_count = 0
        
        # Per-component state
        self.components = {
            'engine': {
                'sensors': {},
                'anomaly': False,
                'reconstruction_error': 0.0,
                'threshold': 0.0,
                'pca_coords': [0, 0, 0]
            },
            'hydraulic': {
                'sensors': {},
                'anomaly': False,
                'reconstruction_error': 0.0,
                'threshold': 0.0,
                'pca_coords': [0, 0, 0]
            },
            'wheels': {
                'sensors': {},
                'anomaly': False,
                'reconstruction_error': 0.0,
                'threshold': 0.0,
                'pca_coords': [0, 0, 0]
            },
            'chassis': {
                'sensors': {},
                'anomaly': False,
                'reconstruction_error': 0.0,
                'threshold': 0.0,
                'pca_coords': [0, 0, 0]
            }
        }
        
        # Global anomaly state (any component anomalous)
        self.is_global_anomaly = False
        
        # All sensor readings (flattened)
        self.all_sensors = {}
        self.anomalous_sensors = []
        
        # Time series buffers (shared across all views)
        self.time_series = {
            'timestamps': [],
            'engine_temp': [],
            'oil_pressure': [],
            'fuel_rate': [],
            'hydraulic_pressure': [],
            'load': [],
            'vibration': [],
            'brake_temp': [],
            'speed': [],
            'max_points': 250
        }
        
        # Injection state
        self.inject_anomaly = False
        self.active_failures = []
        self.failure_locked = False
    
    def update(self, data):
        """Thread-safe state update."""
        with self.lock:
            self.timestamp = datetime.now().isoformat()
            self.tick_count += 1
            
            # Update component states
            for comp_name, comp_data in data.get('components', {}).items():
                if comp_name in self.components:
                    self.components[comp_name].update(comp_data)
            
            # Update global state
            self.is_global_anomaly = data.get('is_global_anomaly', False)
            self.all_sensors = data.get('all_sensors', {})
            self.anomalous_sensors = data.get('anomalous_sensors', [])
            
            # Update time series
            ts = data.get('time_series', {})
            for key in self.time_series:
                if key in ts and key != 'max_points':
                    self.time_series[key] = ts[key]
    
    def get_state(self):
        """Thread-safe state read."""
        with self.lock:
            return {
                'timestamp': self.timestamp,
                'tick_count': self.tick_count,
                'components': dict(self.components),
                'is_global_anomaly': self.is_global_anomaly,
                'all_sensors': dict(self.all_sensors),
                'anomalous_sensors': list(self.anomalous_sensors),
                'time_series': dict(self.time_series),
                'inject_anomaly': self.inject_anomaly,
                'active_failures': list(self.active_failures)
            }
    
    def get_component_state(self, component):
        """Get state for a specific component."""
        with self.lock:
            if component not in self.components:
                return None
            return {
                'timestamp': self.timestamp,
                'tick_count': self.tick_count,
                'component': component,
                **self.components[component],
                'inject_anomaly': self.inject_anomaly,
                'is_failing': component in self.active_failures
            }

# Initialize shared state
shared_state = SharedState()

# ============================================================
# COMPONENT CONFIGURATION
# ============================================================

# Component → Sensor mapping (for data generation)
COMPONENT_SENSORS = {
    'engine': {
        'sensors': ['engine_temp_C', 'oil_pressure_bar', 'fuel_rate_lph'],
        'display_names': {
            'engine_temp_C': 'ENGINE TEMP °C',
            'oil_pressure_bar': 'OIL PRESSURE BAR',
            'fuel_rate_lph': 'FUEL RATE L/H'
        },
        'indices': [7, 5, 2]  # In main sensor array
    },
    'hydraulic': {
        'sensors': ['hydraulic_pressure_bar', 'load_tons'],
        'display_names': {
            'hydraulic_pressure_bar': 'HYDRAULIC PRESSURE BAR',
            'load_tons': 'LOAD TONS'
        },
        'indices': [6, 1]
    },
    'wheels': {
        'sensors': ['vibration_mm_s', 'brake_temp_C', 'speed_kmph'],
        'display_names': {
            'vibration_mm_s': 'VIBRATION MM/S',
            'brake_temp_C': 'BRAKE TEMP °C',
            'speed_kmph': 'SPEED KM/H'
        },
        'indices': [4, 3, 0]
    },
    'chassis': {
        'sensors': ['vibration_mm_s', 'load_tons'],
        'display_names': {
            'vibration_mm_s': 'VIBRATION MM/S',
            'load_tons': 'LOAD TONS'
        },
        'indices': [4, 1]
    }
}

# Failure probability weights
COMPONENT_FAILURE_WEIGHTS = {
    'engine': 0.40,
    'hydraulic': 0.30,
    'wheels': 0.20,
    'chassis': 0.10
}

# ============================================================
# EXPLAINABLE AI (XAI) - FEATURE CONTRIBUTION ANALYSIS
# ============================================================

# Component feature names for explanations
COMPONENT_FEATURE_NAMES = {
    'engine': ['engine_temperature', 'oil_pressure', 'fuel_rate'],
    'hydraulic': ['hydraulic_pressure', 'fluid_temp'],
    'wheels': ['wheel_vibration', 'brake_temperature', 'wheel_speed'],
    'chassis': ['chassis_vibration', 'load_weight']
}

# Human-readable feature descriptions
FEATURE_DESCRIPTIONS = {
    'engine_temperature': 'engine temperature',
    'oil_pressure': 'oil pressure',
    'fuel_rate': 'fuel consumption rate',
    'hydraulic_pressure': 'hydraulic pressure',
    'fluid_temp': 'hydraulic fluid temperature',
    'wheel_vibration': 'wheel vibration',
    'brake_temperature': 'brake temperature',
    'wheel_speed': 'wheel speed',
    'chassis_vibration': 'structural vibration',
    'load_weight': 'load weight'
}

# Severity adjectives based on contribution percentage
def get_severity_adjective(contribution):
    """Return appropriate adjective based on contribution level."""
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

# Component-specific cause descriptors
COMPONENT_CAUSE_TEMPLATES = {
    'engine': {
        'engine_temperature': ['overheating condition', 'thermal stress', 'cooling system inefficiency'],
        'oil_pressure': ['lubrication system stress', 'potential oil pump degradation', 'viscosity issues'],
        'fuel_rate': ['fuel injection irregularity', 'combustion inefficiency', 'fuel system imbalance']
    },
    'hydraulic': {
        'hydraulic_pressure': ['possible blockage or valve malfunction', 'pump performance degradation', 'seal integrity issues'],
        'fluid_temp': ['fluid overheating', 'cooling circuit inefficiency', 'excessive system load']
    },
    'wheels': {
        'wheel_vibration': ['imbalance or surface irregularities', 'bearing wear', 'alignment deviation'],
        'brake_temperature': ['brake system overheating', 'friction material degradation', 'caliper binding'],
        'wheel_speed': ['speed sensor anomaly', 'drivetrain irregularity', 'traction control engagement']
    },
    'chassis': {
        'chassis_vibration': ['structural resonance', 'mounting point stress', 'frame fatigue indication'],
        'load_weight': ['excessive payload', 'load distribution imbalance', 'suspension overload']
    }
}

def calculate_feature_contributions(X_scaled, X_pred, feature_names):
    """
    Calculate per-feature reconstruction error contributions.
    
    Returns:
        list: Sorted list of {feature, contribution, error} dicts
    """
    # Compute per-feature squared error
    feature_errors = np.square(X_scaled.flatten() - X_pred.flatten())
    total_error = np.sum(feature_errors)
    
    if total_error == 0:
        return []
    
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

def generate_explanation_text(component, contributions):
    """
    Generate natural language explanation from feature contributions.
    
    Args:
        component: Component name (engine, hydraulic, wheels, chassis)
        contributions: Sorted list of feature contributions
    
    Returns:
        str: Human-readable explanation
    """
    if not contributions:
        return "Anomaly detected but contribution analysis unavailable."
    
    component_names = {
        'engine': 'Engine',
        'hydraulic': 'Hydraulic system',
        'wheels': 'Wheel assembly',
        'chassis': 'Chassis'
    }
    
    comp_display = component_names.get(component, component.title())
    
    # Primary cause (highest contributor)
    primary = contributions[0]
    primary_desc = FEATURE_DESCRIPTIONS.get(primary['feature'], primary['feature'])
    primary_severity = get_severity_adjective(primary['contribution'])
    
    # Get specific cause template
    cause_templates = COMPONENT_CAUSE_TEMPLATES.get(component, {})
    specific_cause = cause_templates.get(primary['feature'], ['abnormal readings'])[0]
    
    # Build explanation
    explanation_parts = []
    
    # Opening statement
    if primary['contribution'] >= 50:
        explanation_parts.append(
            f"{comp_display} anomaly detected primarily due to {primary_severity} elevated {primary_desc}, "
            f"contributing {primary['contribution']:.0f}% of the total deviation. "
            f"This indicates {specific_cause}."
        )
    else:
        explanation_parts.append(
            f"{comp_display} anomaly detected with {primary_desc} as the leading factor "
            f"({primary['contribution']:.0f}% contribution), suggesting {specific_cause}."
        )
    
    # Secondary factors (if significant)
    if len(contributions) > 1 and contributions[1]['contribution'] >= 15:
        secondary = contributions[1]
        sec_desc = FEATURE_DESCRIPTIONS.get(secondary['feature'], secondary['feature'])
        sec_cause = cause_templates.get(secondary['feature'], ['elevated readings'])[0]
        explanation_parts.append(
            f"Elevated {sec_desc} ({secondary['contribution']:.0f}%) further indicates {sec_cause}."
        )
    
    # Minor factors summary
    minor_factors = [c for c in contributions[1:] if 5 <= c['contribution'] < 15]
    if minor_factors:
        minor_names = [FEATURE_DESCRIPTIONS.get(c['feature'], c['feature']) for c in minor_factors]
        if len(minor_names) == 1:
            explanation_parts.append(f"Minor deviation observed in {minor_names[0]}.")
        elif len(minor_names) > 1:
            explanation_parts.append(f"Minor deviations observed in {', '.join(minor_names[:-1])} and {minor_names[-1]}.")
    
    return " ".join(explanation_parts)

# Failure scenarios
FAILURE_SCENARIOS = [
    (['engine'], 0.30),
    (['hydraulic'], 0.22),
    (['wheels'], 0.13),
    (['chassis'], 0.05),
    (['engine', 'hydraulic'], 0.12),
    (['wheels', 'chassis'], 0.10),
    (['hydraulic', 'wheels'], 0.08)
]

# ============================================================
# LOAD ALL MODELS AT STARTUP
# ============================================================
logger.info("=" * 60)
logger.info("LOADING ALL MODEL ARTIFACTS")
logger.info("=" * 60)

# Main model (for global anomaly detection)
main_model = None
main_scaler = None
main_pca = None
main_threshold = 0.5
main_stats = None

# Component-specific models
component_models = {}

try:
    # Load main model
    logger.info("[MAIN MODEL]")
    main_model = keras.models.load_model(os.path.join(BASE_DIR, 'autoencoder_model.h5'), compile=False)
    main_scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    main_pca = joblib.load(os.path.join(BASE_DIR, 'pca.pkl'))
    main_threshold = joblib.load(os.path.join(BASE_DIR, 'threshold.pkl'))
    main_stats = joblib.load(os.path.join(BASE_DIR, 'data_stats.pkl'))
    logger.info(f"  ✅ Loaded main autoencoder (threshold: {main_threshold:.4f})")
except Exception as e:
    logger.warning(f"  ⚠️ Main model not loaded: {e}")
    main_stats = {
        'mean': [35, 55, 45, 120, 2.5, 4.5, 210, 85],
        'std': [5, 10, 8, 15, 0.5, 0.8, 20, 10],
        'sensor_cols': ['Speed_kmph', 'Load_tons', 'Fuel_Rate_Lph', 'Brake_Temp_C',
                        'Vibration_mm_s', 'Oil_Pressure_bar', 'Hydraulic_Pressure_bar', 'Engine_Temp_C']
    }

# Load component models
for component in ['engine', 'hydraulic', 'wheels', 'chassis']:
    logger.info(f"[{component.upper()} MODEL]")
    try:
        model_path = os.path.join(BASE_DIR, f'{component}_autoencoder.h5')
        scaler_path = os.path.join(BASE_DIR, f'{component}_scaler.pkl')
        pca_path = os.path.join(BASE_DIR, f'{component}_pca.pkl')
        threshold_path = os.path.join(BASE_DIR, f'{component}_threshold.pkl')
        stats_path = os.path.join(BASE_DIR, f'{component}_stats.pkl')
        
        component_models[component] = {
            'model': keras.models.load_model(model_path, compile=False),
            'scaler': joblib.load(scaler_path),
            'pca': joblib.load(pca_path),
            'threshold': joblib.load(threshold_path),
            'stats': joblib.load(stats_path)
        }
        logger.info(f"  ✅ Loaded (threshold: {component_models[component]['threshold']:.4f})")
    except Exception as e:
        logger.warning(f"  ⚠️ Not loaded: {e}")
        component_models[component] = None

logger.info("=" * 60)
logger.info("MODEL LOADING COMPLETE")
logger.info("=" * 60)

# Sensor display name mapping
SENSOR_DISPLAY_NAMES = {
    'Speed_kmph': 'SPEED KM/H',
    'Load_tons': 'LOAD TONS',
    'Fuel_Rate_Lph': 'FUEL RATE L/H',
    'Brake_Temp_C': 'BRAKE TEMP °C',
    'Vibration_mm_s': 'VIBRATION MM/S',
    'Oil_Pressure_bar': 'OIL PRESSURE BAR',
    'Hydraulic_Pressure_bar': 'HYDRAULIC PRESSURE BAR',
    'Engine_Temp_C': 'ENGINE TEMP °C'
}

# Create case-insensitive lookup for display names
SENSOR_DISPLAY_NAMES_LOWER = {k.lower(): v for k, v in SENSOR_DISPLAY_NAMES.items()}

import random

def select_failure_scenario():
    """Select a realistic failure scenario based on weighted probabilities."""
    rand = random.random()
    cumulative = 0.0
    for components, probability in FAILURE_SCENARIOS:
        cumulative += probability
        if rand <= cumulative:
            return components.copy()
    return ['engine']

# ============================================================
# ROUTES - PAGE RENDERING (Instant Response)
# ============================================================

@app.route('/')
def index():
    """Main dashboard - instant render, data loaded via AJAX."""
    return render_template('index.html')

@app.route('/component/<component_name>')
def component_interface(component_name):
    """
    Component-specific interface.
    Renders instantly - data synchronized via shared state.
    """
    valid_components = ['engine', 'hydraulic', 'wheels', 'chassis']
    if component_name not in valid_components:
        return "Component not found", 404
    
    # Render component template (instant - no data fetch)
    return render_template(f'{component_name}.html', component=component_name)

# ============================================================
# API ROUTES - DATA ENDPOINTS
# ============================================================

@app.route('/toggle_injection', methods=['POST'])
def toggle_injection():
    """Toggle anomaly injection state."""
    with shared_state.lock:
        shared_state.inject_anomaly = not shared_state.inject_anomaly
        current_state = shared_state.inject_anomaly
        
        if current_state:
            shared_state.active_failures = select_failure_scenario()
            shared_state.failure_locked = True
            logger.info(f"[INJECTION ON] Failing: {shared_state.active_failures}")
        else:
            shared_state.active_failures = []
            shared_state.failure_locked = False
            logger.info("[INJECTION OFF] All normal")
    
    return jsonify({
        'inject_anomaly': current_state,
        'failed_components': shared_state.active_failures
    })

@app.route('/get_injection_state')
def get_injection_state():
    """Get current injection state."""
    return jsonify({
        'inject_anomaly': shared_state.inject_anomaly,
        'failed_components': shared_state.active_failures,
        'failure_probabilities': COMPONENT_FAILURE_WEIGHTS
    })

@app.route('/simulate_data')
def simulate_data():
    """
    Generate and analyze simulated sensor data.
    Updates the SHARED STATE - all views read from here.
    """
    # Get injection state
    inject = shared_state.inject_anomaly
    active_failures = shared_state.active_failures.copy()
    
    # Generate base sensor data
    means = np.array(main_stats['mean'])
    stds = np.array(main_stats['std'])
    raw_data = means + np.random.randn(len(means)) * stds * 0.3
    
    # Track spiked sensors
    spiked_sensor_indices = set()
    
    # Apply component-based failure injection
    if inject and active_failures:
        for component in active_failures:
            if component in COMPONENT_SENSORS:
                for sensor_idx in COMPONENT_SENSORS[component]['indices']:
                    spike_magnitude = 8 + np.random.rand() * 4
                    raw_data[sensor_idx] = means[sensor_idx] + spike_magnitude * stds[sensor_idx]
                    raw_data[sensor_idx] += np.random.randn() * stds[sensor_idx] * 0.3
                    spiked_sensor_indices.add(sensor_idx)
    
    raw_data = np.maximum(raw_data, 0.1)
    
    # Create sensor readings
    sensor_cols = main_stats['sensor_cols']
    all_sensors = {}
    anomalous_sensors = []
    
    for i, col in enumerate(sensor_cols):
        # Case-insensitive lookup for display name
        display_name = SENSOR_DISPLAY_NAMES_LOWER.get(col.lower(), col)
        value = float(raw_data[i])
        all_sensors[display_name] = round(value, 2)
        
        if i in spiked_sensor_indices and value > means[i] + 3.0 * stds[i]:
            anomalous_sensors.append(display_name)
    
    # Global anomaly detection (main model)
    global_anomaly = False
    global_recon_error = 0.0
    global_pca_coords = [0, 0, 0]
    
    # DEBUG: Check model/scaler status
    model_loaded = main_model is not None
    scaler_loaded = main_scaler is not None
    
    if model_loaded and scaler_loaded:
        try:
            # Step 1: Scale the raw data
            X_input = raw_data.reshape(1, -1)
            X_scaled = main_scaler.transform(X_input)
            
            # Step 2: Get model prediction (reconstruction)
            X_pred = main_model.predict(X_scaled, verbose=0)
            
            # Step 3: Calculate reconstruction error CORRECTLY
            diff = X_scaled - X_pred
            squared_diff = np.square(diff)
            global_recon_error = float(np.mean(squared_diff))
            
            # Step 4: Determine anomaly status
            global_anomaly = global_recon_error > main_threshold
            
            # Step 5: PCA coordinates
            if main_pca is not None:
                global_pca_coords = main_pca.transform(X_scaled)[0].tolist()
            
            # DEBUG LOG (every 200 ticks to reduce spam)
            if shared_state.tick_count % 200 == 0:
                logger.debug(f"[TICK {shared_state.tick_count}] Injection={inject}, Failures={active_failures}, Error={global_recon_error:.6f}, Threshold={main_threshold:.6f}, Anomaly={global_anomaly}")
                
        except Exception as e:
            logger.error(f"Main model error: {e}")
            import traceback
            traceback.print_exc()
            global_anomaly = inject
    else:
        logger.warning(f"[WARNING] Model loaded: {model_loaded}, Scaler loaded: {scaler_loaded}")
        global_anomaly = inject
        global_pca_coords = [np.random.randn() * 5 for _ in range(3)]
    
    # Per-component anomaly detection
    component_states = {}
    affected_components = {'engine': False, 'hydraulic': False, 'chassis': False, 'wheels': False}
    
    for comp_name, comp_config in COMPONENT_SENSORS.items():
        # Extract component-specific sensor values
        comp_sensors = {}
        comp_raw = []
        
        for sensor_name in comp_config['sensors']:
            display_name = comp_config['display_names'][sensor_name]
            # Find the value in all_sensors
            if display_name in all_sensors:
                comp_sensors[display_name] = all_sensors[display_name]
                comp_raw.append(all_sensors[display_name])
        
        # Component-specific anomaly detection
        comp_anomaly = False
        comp_recon_error = 0.0
        comp_threshold = 0.0
        comp_pca = [0, 0, 0]
        
        # Skip if no sensor values found (defensive)
        if len(comp_raw) == 0:
            component_states[comp_name] = {
                'sensors': comp_sensors,
                'anomaly': False,
                'reconstruction_error': 0.0,
                'threshold': 0.0,
                'pca_coords': [0, 0, 0]
            }
            continue
        
        if component_models.get(comp_name) is not None:
            try:
                model_info = component_models[comp_name]
                comp_threshold = float(model_info['threshold'])
                
                # Scale and predict using CORRECT formula
                X_comp = np.array(comp_raw, dtype=np.float32).reshape(1, -1)
                X_comp_scaled = model_info['scaler'].transform(X_comp)
                X_comp_pred = model_info['model'].predict(X_comp_scaled, verbose=0)
                
                # Calculate reconstruction error: mean((X_scaled - X_reconstructed)^2)
                comp_diff = X_comp_scaled - X_comp_pred
                comp_recon_error = float(np.mean(np.square(comp_diff)))
                comp_anomaly = comp_recon_error > comp_threshold
                
                # PCA coordinates
                n_pca = model_info['pca'].n_components_
                pca_result = model_info['pca'].transform(X_comp_scaled)[0].tolist()
                comp_pca = pca_result + [0] * (3 - len(pca_result))
                
                # XAI: Calculate feature contributions (for anomaly explanation)
                comp_feature_names = COMPONENT_FEATURE_NAMES.get(comp_name, [])
                comp_contributions = calculate_feature_contributions(
                    X_comp_scaled, X_comp_pred, comp_feature_names
                )
                
            except Exception as e:
                logger.error(f"[{comp_name}] Model error: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: use injection state
                comp_anomaly = comp_name in active_failures if inject else False
                comp_contributions = []
        else:
            comp_anomaly = comp_name in active_failures if inject else False
            comp_contributions = []
        
        # Override with injection state if active
        if inject and comp_name in active_failures:
            comp_anomaly = True
        
        affected_components[comp_name] = comp_anomaly
        
        # Generate explanation only when anomaly is detected
        root_cause_numeric = []
        root_cause_text = ""
        if comp_anomaly and comp_contributions:
            root_cause_numeric = comp_contributions
            root_cause_text = generate_explanation_text(comp_name, comp_contributions)
        
        component_states[comp_name] = {
            'sensors': comp_sensors,
            'anomaly': comp_anomaly,
            'reconstruction_error': round(comp_recon_error, 4),
            'threshold': round(comp_threshold, 4),
            'pca_coords': comp_pca,
            'root_cause_numeric': root_cause_numeric,
            'root_cause_text': root_cause_text
        }
    
    # Update time series
    ts = shared_state.time_series
    # Use global tick_count as timestamp (NOT len() which stays at max_points after buffer fills)
    current_tick = shared_state.tick_count
    
    ts['timestamps'].append(current_tick)
    ts['engine_temp'].append(float(raw_data[7]))
    ts['oil_pressure'].append(float(raw_data[5]))
    ts['fuel_rate'].append(float(raw_data[2]))
    ts['hydraulic_pressure'].append(float(raw_data[6]))
    ts['load'].append(float(raw_data[1]))
    ts['vibration'].append(float(raw_data[4]))
    ts['brake_temp'].append(float(raw_data[3]))
    ts['speed'].append(float(raw_data[0]))
    
    # Trim to max points
    max_pts = ts['max_points']
    for key in ts:
        if key != 'max_points' and isinstance(ts[key], list) and len(ts[key]) > max_pts:
            ts[key] = ts[key][-max_pts:]
    
    # Update shared state
    shared_state.update({
        'components': component_states,
        'is_global_anomaly': global_anomaly,
        'all_sensors': all_sensors,
        'anomalous_sensors': anomalous_sensors,
        'time_series': ts
    })
    
    # Build response (for main dashboard)
    # 3D visualization with actual sensor values (not PCA)
    # X: Engine Temperature, Y: Hydraulic Pressure, Z: Vibration
    sensor_3d_coords = [
        float(raw_data[7]),  # Engine Temp (°C)
        float(raw_data[6]),  # Hydraulic Pressure (bar)
        float(raw_data[4])   # Vibration (mm/s)
    ]
    
    response = {
        'timestamp': shared_state.timestamp,
        'sensor_readings': all_sensors,
        'anomalous_parameters': anomalous_sensors,
        'reconstruction_error': round(global_recon_error, 4),
        'threshold': round(float(main_threshold), 4),
        'is_anomaly': bool(global_anomaly),
        'pca_coords': global_pca_coords,
        'sensor_3d_coords': sensor_3d_coords,
        'affected_components': affected_components,
        'failed_components': active_failures,
        'failure_probabilities': COMPONENT_FAILURE_WEIGHTS,
        'inject_active': inject,
        'time_series': {
            'timestamps': ts['timestamps'][-100:],
            'engine_temp': ts['engine_temp'][-100:],
            'hydraulic_pressure': ts['hydraulic_pressure'][-100:],
            'vibration': ts['vibration'][-100:],
            'speed': ts['speed'][-100:]
        },
        'component_states': component_states
    }
    
    return jsonify(response)

@app.route('/api/component/<component_name>')
def api_component_data(component_name):
    """
    API endpoint for component-specific data.
    TRIGGERS data generation to ensure SharedState is fresh,
    then returns component-specific slice.
    """
    valid_components = ['engine', 'hydraulic', 'wheels', 'chassis']
    if component_name not in valid_components:
        return jsonify({'error': 'Invalid component'}), 404
    
    # CRITICAL: Generate fresh data first!
    # This ensures SharedState.components has sensor data
    simulate_data()
    
    # Now read from updated shared state (thread-safe)
    state = shared_state.get_state()
    comp_state = state['components'].get(component_name, {})
    
    # Get component-specific time series
    ts = state['time_series']
    comp_time_series = {'timestamps': ts['timestamps'][-100:]}
    
    # Add component-specific sensors to time series
    if component_name == 'engine':
        comp_time_series['engine_temp'] = ts['engine_temp'][-100:]
        comp_time_series['oil_pressure'] = ts['oil_pressure'][-100:]
        comp_time_series['fuel_rate'] = ts['fuel_rate'][-100:]
    elif component_name == 'hydraulic':
        comp_time_series['hydraulic_pressure'] = ts['hydraulic_pressure'][-100:]
        comp_time_series['load'] = ts['load'][-100:]
    elif component_name == 'wheels':
        comp_time_series['vibration'] = ts['vibration'][-100:]
        comp_time_series['brake_temp'] = ts['brake_temp'][-100:]
        comp_time_series['speed'] = ts['speed'][-100:]
    elif component_name == 'chassis':
        comp_time_series['vibration'] = ts['vibration'][-100:]
        comp_time_series['load'] = ts['load'][-100:]
    
    return jsonify({
        'timestamp': state['timestamp'],
        'component': component_name,
        'sensors': comp_state.get('sensors', {}),
        'anomaly': bool(comp_state.get('anomaly', False)),
        'reconstruction_error': float(comp_state.get('reconstruction_error', 0)),
        'threshold': float(comp_state.get('threshold', 0)),
        'pca_coords': comp_state.get('pca_coords', [0, 0, 0]),
        'time_series': comp_time_series,
        'inject_active': state['inject_anomaly'],
        'is_failing': component_name in state['active_failures']
    })

@app.route('/api/live_state')
def api_live_state():
    """
    UNIFIED LIVE STATE ENDPOINT
    ==========================
    This is THE endpoint that ALL dashboards must call.
    It triggers data generation AND returns complete state.
    
    Response structure:
    {
        "timestamp": "...",
        "engine": { "sensors": {...}, "anomaly": true/false, ... },
        "hydraulic": { "sensors": {...}, "anomaly": true/false, ... },
        "wheels": { "sensors": {...}, "anomaly": true/false, ... },
        "chassis": { "sensors": {...}, "anomaly": true/false, ... },
        "global": { ... }
    }
    """
    # First, generate fresh data by calling simulate_data logic internally
    # This ensures SharedState is always up-to-date
    sim_response = simulate_data()
    sim_data = sim_response.get_json()
    
    # Get the updated shared state
    state = shared_state.get_state()
    ts = state['time_series']
    
    # Build component-specific responses
    response = {
        'timestamp': state['timestamp'],
        'inject_active': state['inject_anomaly'],
        'failed_components': state['active_failures'],
        
        # Engine component
        'engine': {
            'sensors': state['components']['engine'].get('sensors', {}),
            'anomaly': bool(state['components']['engine'].get('anomaly', False)),
            'reconstruction_error': float(state['components']['engine'].get('reconstruction_error', 0)),
            'threshold': float(state['components']['engine'].get('threshold', 0)),
            'pca_coords': state['components']['engine'].get('pca_coords', [0, 0, 0]),
            'root_cause_numeric': state['components']['engine'].get('root_cause_numeric', []),
            'root_cause_text': state['components']['engine'].get('root_cause_text', ''),
            'time_series': {
                'timestamps': ts['timestamps'][-100:],
                'engine_temp': ts['engine_temp'][-100:],
                'oil_pressure': ts['oil_pressure'][-100:],
                'fuel_rate': ts['fuel_rate'][-100:]
            }
        },
        
        # Hydraulic component  
        'hydraulic': {
            'sensors': state['components']['hydraulic'].get('sensors', {}),
            'anomaly': bool(state['components']['hydraulic'].get('anomaly', False)),
            'reconstruction_error': float(state['components']['hydraulic'].get('reconstruction_error', 0)),
            'threshold': float(state['components']['hydraulic'].get('threshold', 0)),
            'pca_coords': state['components']['hydraulic'].get('pca_coords', [0, 0, 0]),
            'root_cause_numeric': state['components']['hydraulic'].get('root_cause_numeric', []),
            'root_cause_text': state['components']['hydraulic'].get('root_cause_text', ''),
            'time_series': {
                'timestamps': ts['timestamps'][-100:],
                'hydraulic_pressure': ts['hydraulic_pressure'][-100:],
                'load': ts['load'][-100:]
            }
        },
        
        # Wheels component
        'wheels': {
            'sensors': state['components']['wheels'].get('sensors', {}),
            'anomaly': bool(state['components']['wheels'].get('anomaly', False)),
            'reconstruction_error': float(state['components']['wheels'].get('reconstruction_error', 0)),
            'threshold': float(state['components']['wheels'].get('threshold', 0)),
            'pca_coords': state['components']['wheels'].get('pca_coords', [0, 0, 0]),
            'root_cause_numeric': state['components']['wheels'].get('root_cause_numeric', []),
            'root_cause_text': state['components']['wheels'].get('root_cause_text', ''),
            'time_series': {
                'timestamps': ts['timestamps'][-100:],
                'vibration': ts['vibration'][-100:],
                'brake_temp': ts['brake_temp'][-100:],
                'speed': ts['speed'][-100:]
            }
        },
        
        # Chassis component
        'chassis': {
            'sensors': state['components']['chassis'].get('sensors', {}),
            'anomaly': bool(state['components']['chassis'].get('anomaly', False)),
            'reconstruction_error': float(state['components']['chassis'].get('reconstruction_error', 0)),
            'threshold': float(state['components']['chassis'].get('threshold', 0)),
            'pca_coords': state['components']['chassis'].get('pca_coords', [0, 0, 0]),
            'root_cause_numeric': state['components']['chassis'].get('root_cause_numeric', []),
            'root_cause_text': state['components']['chassis'].get('root_cause_text', ''),
            'time_series': {
                'timestamps': ts['timestamps'][-100:],
                'vibration': ts['vibration'][-100:],
                'load': ts['load'][-100:]
            }
        },
        
        # Global state (for main dashboard)
        'global': {
            'sensor_readings': sim_data.get('sensor_readings', {}),
            'anomalous_parameters': sim_data.get('anomalous_parameters', []),
            'reconstruction_error': sim_data.get('reconstruction_error', 0),
            'threshold': sim_data.get('threshold', 0),
            'is_anomaly': sim_data.get('is_anomaly', False),
            'pca_coords': sim_data.get('pca_coords', [0, 0, 0]),
            'affected_components': sim_data.get('affected_components', {}),
            'time_series': sim_data.get('time_series', {})
        }
    }
    
    return jsonify(response)

@app.route('/api/state')
def api_full_state():
    """Get complete shared state for debugging/monitoring."""
    state = shared_state.get_state()
    # Convert numpy booleans to Python booleans for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        return obj
    return jsonify(convert_numpy(state))

@app.route('/reset_buffer', methods=['POST'])
def reset_buffer():
    """Reset time series buffers."""
    with shared_state.lock:
        for key in shared_state.time_series:
            if key != 'max_points':
                shared_state.time_series[key] = []
    return jsonify({'status': 'ok'})

@app.route('/get_failure_probabilities')
def get_failure_probabilities():
    """Return failure probability weights."""
    return jsonify({
        'probabilities': COMPONENT_FAILURE_WEIGHTS,
        'description': {
            'engine': 'High thermal/mechanical stress',
            'hydraulic': 'Fluid system leaks and seal wear',
            'wheels': 'Brake wear and bearing issues',
            'chassis': 'Structural - rarely fails'
        }
    })

if __name__ == '__main__':
    # Use PORT environment variable for Render deployment
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    logger.info(f"Starting server on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port, threaded=True, use_reloader=False)
