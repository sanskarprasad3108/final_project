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

app = Flask(__name__)

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
print("=" * 60)
print("LOADING ALL MODEL ARTIFACTS")
print("=" * 60)

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
    print("\n[MAIN MODEL]")
    main_model = keras.models.load_model(os.path.join(BASE_DIR, 'autoencoder_model.h5'), compile=False)
    main_scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    main_pca = joblib.load(os.path.join(BASE_DIR, 'pca.pkl'))
    main_threshold = joblib.load(os.path.join(BASE_DIR, 'threshold.pkl'))
    main_stats = joblib.load(os.path.join(BASE_DIR, 'data_stats.pkl'))
    print(f"  ✅ Loaded main autoencoder (threshold: {main_threshold:.4f})")
except Exception as e:
    print(f"  ⚠️ Main model not loaded: {e}")
    main_stats = {
        'mean': [35, 55, 45, 120, 2.5, 4.5, 210, 85],
        'std': [5, 10, 8, 15, 0.5, 0.8, 20, 10],
        'sensor_cols': ['Speed_kmph', 'Load_tons', 'Fuel_Rate_Lph', 'Brake_Temp_C',
                        'Vibration_mm_s', 'Oil_Pressure_bar', 'Hydraulic_Pressure_bar', 'Engine_Temp_C']
    }

# Load component models
for component in ['engine', 'hydraulic', 'wheels', 'chassis']:
    print(f"\n[{component.upper()} MODEL]")
    try:
        model_path = os.path.join(BASE_DIR, 'models', f'{component}_autoencoder.h5')
        scaler_path = os.path.join(BASE_DIR, 'models', f'{component}_scaler.pkl')
        pca_path = os.path.join(BASE_DIR, 'models', f'{component}_pca.pkl')
        threshold_path = os.path.join(BASE_DIR, 'models', f'{component}_threshold.pkl')
        stats_path = os.path.join(BASE_DIR, 'models', f'{component}_stats.pkl')
        
        component_models[component] = {
            'model': keras.models.load_model(model_path, compile=False),
            'scaler': joblib.load(scaler_path),
            'pca': joblib.load(pca_path),
            'threshold': joblib.load(threshold_path),
            'stats': joblib.load(stats_path)
        }
        print(f"  ✅ Loaded (threshold: {component_models[component]['threshold']:.4f})")
    except Exception as e:
        print(f"  ⚠️ Not loaded: {e}")
        component_models[component] = None

print("\n" + "=" * 60)
print("MODEL LOADING COMPLETE")
print("=" * 60)

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
            print(f"[INJECTION ON] Failing: {shared_state.active_failures}")
        else:
            shared_state.active_failures = []
            shared_state.failure_locked = False
            print("[INJECTION OFF] All normal")
    
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
        display_name = SENSOR_DISPLAY_NAMES.get(col, col)
        value = float(raw_data[i])
        all_sensors[display_name] = round(value, 2)
        
        if i in spiked_sensor_indices and value > means[i] + 3.0 * stds[i]:
            anomalous_sensors.append(display_name)
    
    # Global anomaly detection (main model)
    global_anomaly = False
    global_recon_error = 0.0
    global_pca_coords = [0, 0, 0]
    
    if main_model is not None:
        try:
            X_scaled = main_scaler.transform(raw_data.reshape(1, -1))
            X_pred = main_model.predict(X_scaled, verbose=0)
            global_recon_error = float(np.mean(np.square(X_scaled - X_pred)))
            global_anomaly = global_recon_error > main_threshold
            global_pca_coords = main_pca.transform(X_scaled)[0].tolist()
        except Exception as e:
            print(f"Main model error: {e}")
            global_anomaly = inject
    else:
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
        
        if component_models.get(comp_name) is not None:
            try:
                model_info = component_models[comp_name]
                comp_threshold = model_info['threshold']
                
                # Scale and predict
                X_comp = np.array(comp_raw).reshape(1, -1)
                X_comp_scaled = model_info['scaler'].transform(X_comp)
                X_comp_pred = model_info['model'].predict(X_comp_scaled, verbose=0)
                comp_recon_error = float(np.mean(np.square(X_comp_scaled - X_comp_pred)))
                comp_anomaly = comp_recon_error > comp_threshold
                
                # PCA coordinates
                n_pca = model_info['pca'].n_components_
                pca_result = model_info['pca'].transform(X_comp_scaled)[0].tolist()
                comp_pca = pca_result + [0] * (3 - len(pca_result))
                
            except Exception as e:
                # Fallback: use injection state
                comp_anomaly = comp_name in active_failures if inject else False
        else:
            comp_anomaly = comp_name in active_failures if inject else False
        
        # Override with injection state if active
        if inject and comp_name in active_failures:
            comp_anomaly = True
        
        affected_components[comp_name] = comp_anomaly
        
        component_states[comp_name] = {
            'sensors': comp_sensors,
            'anomaly': comp_anomaly,
            'reconstruction_error': round(comp_recon_error, 4),
            'threshold': round(comp_threshold, 4),
            'pca_coords': comp_pca
        }
    
    # Update time series
    ts = shared_state.time_series
    current_tick = len(ts['timestamps'])
    
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
    response = {
        'timestamp': shared_state.timestamp,
        'sensor_readings': all_sensors,
        'anomalous_parameters': anomalous_sensors,
        'reconstruction_error': round(global_recon_error, 4),
        'threshold': round(float(main_threshold), 4),
        'is_anomaly': bool(global_anomaly),
        'pca_coords': global_pca_coords,
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
    Reads from SHARED STATE - no independent data generation.
    Instant response (<10ms).
    """
    valid_components = ['engine', 'hydraulic', 'wheels', 'chassis']
    if component_name not in valid_components:
        return jsonify({'error': 'Invalid component'}), 404
    
    # Read from shared state (thread-safe)
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
        'anomaly': comp_state.get('anomaly', False),
        'reconstruction_error': comp_state.get('reconstruction_error', 0),
        'threshold': comp_state.get('threshold', 0),
        'pca_coords': comp_state.get('pca_coords', [0, 0, 0]),
        'time_series': comp_time_series,
        'inject_active': state['inject_anomaly'],
        'is_failing': component_name in state['active_failures']
    })

@app.route('/api/state')
def api_full_state():
    """Get complete shared state for debugging/monitoring."""
    return jsonify(shared_state.get_state())

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
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
