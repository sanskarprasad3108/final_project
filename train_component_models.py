"""
Train Component-Specific Autoencoder Models
============================================
Trains ONE autoencoder per component dataset.
Each model only sees sensors for its specific component.

Models are saved separately in the 'models/' directory:
- models/engine_autoencoder.h5
- models/hydraulic_autoencoder.h5
- models/wheels_autoencoder.h5
- models/chassis_autoencoder.h5

Each model has its own scaler, PCA, and threshold.

CRITICAL RULES:
- NEVER merge datasets
- NEVER cross-train sensors
- Each model operates independently
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings
warnings.filterwarnings('ignore')

# Component definitions with sensor columns
COMPONENT_CONFIG = {
    'engine': {
        'sensors': ['engine_temp_C', 'oil_pressure_bar', 'fuel_rate_lph'],
        'encoding_dim': 2,
        'epochs': 100
    },
    'hydraulic': {
        'sensors': ['hydraulic_pressure_bar', 'load_tons'],
        'encoding_dim': 1,
        'epochs': 100
    },
    'wheels': {
        'sensors': ['vibration_mm_s', 'brake_temp_C', 'speed_kmph'],
        'encoding_dim': 2,
        'epochs': 100
    },
    'chassis': {
        'sensors': ['vibration_mm_s', 'load_tons'],
        'encoding_dim': 1,
        'epochs': 100
    }
}

def build_autoencoder(input_dim, encoding_dim):
    """Build autoencoder architecture for given input dimension."""
    
    # Calculate layer sizes
    hidden_dim = max(input_dim * 2, 8)
    
    # Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_dim, activation='relu')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(hidden_dim // 2, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    encoded = layers.Dense(encoding_dim, activation='relu')(x)
    
    # Decoder
    x = layers.Dense(hidden_dim // 2, activation='relu')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    decoded = layers.Dense(input_dim, activation='linear')(x)
    
    # Full autoencoder
    autoencoder = keras.Model(encoder_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def train_component_model(component_name):
    """Train autoencoder for a specific component."""
    
    print(f"\n{'='*60}")
    print(f"TRAINING {component_name.upper()} AUTOENCODER")
    print(f"{'='*60}")
    
    config = COMPONENT_CONFIG[component_name]
    sensors = config['sensors']
    
    # Load component dataset
    dataset_path = f'datasets/{component_name}.csv'
    df = pd.read_csv(dataset_path)
    print(f"Dataset: {dataset_path}")
    print(f"Sensors: {sensors}")
    print(f"Total samples: {len(df)}")
    
    # Split into normal and anomaly
    df_normal = df[df['label'] == 0]
    df_anomaly = df[df['label'] == 1]
    print(f"Normal samples: {len(df_normal)}")
    print(f"Anomaly samples: {len(df_anomaly)}")
    
    # Extract features (only sensors, no timestamp/label)
    X_normal = df_normal[sensors].values
    X_anomaly = df_anomaly[sensors].values if len(df_anomaly) > 0 else None
    
    # Train-test split (80-20 on normal data)
    n_train = int(len(X_normal) * 0.8)
    X_train = X_normal[:n_train]
    X_test = X_normal[n_train:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA for visualization (3 components max, or less for smaller feature sets)
    n_pca = min(3, len(sensors))
    pca = PCA(n_components=n_pca)
    pca.fit(X_train_scaled)
    
    # Build and train autoencoder
    input_dim = len(sensors)
    encoding_dim = config['encoding_dim']
    autoencoder = build_autoencoder(input_dim, encoding_dim)
    
    print(f"\nModel architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Encoding dim: {encoding_dim}")
    
    # Train
    print(f"\nTraining for {config['epochs']} epochs...")
    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=config['epochs'],
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    final_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    print(f"Final training loss: {final_loss:.6f}")
    print(f"Final validation loss: {val_loss:.6f}")
    
    # Calculate threshold on test set (99th percentile of reconstruction error)
    X_test_pred = autoencoder.predict(X_test_scaled, verbose=0)
    test_errors = np.mean(np.square(X_test_scaled - X_test_pred), axis=1)
    threshold = np.percentile(test_errors, 99)
    
    # Also test on anomaly data if available
    if X_anomaly is not None and len(X_anomaly) > 0:
        X_anomaly_scaled = scaler.transform(X_anomaly)
        X_anomaly_pred = autoencoder.predict(X_anomaly_scaled, verbose=0)
        anomaly_errors = np.mean(np.square(X_anomaly_scaled - X_anomaly_pred), axis=1)
        
        print(f"\nReconstruction errors:")
        print(f"  Normal (test): mean={np.mean(test_errors):.4f}, max={np.max(test_errors):.4f}")
        print(f"  Anomaly: mean={np.mean(anomaly_errors):.4f}, min={np.min(anomaly_errors):.4f}")
        
        # Ensure threshold separates normal from anomaly
        if np.max(test_errors) > np.min(anomaly_errors):
            # Overlap exists, use midpoint
            threshold = (np.max(test_errors) + np.min(anomaly_errors)) / 2
    
    print(f"Threshold: {threshold:.6f}")
    
    # Save artifacts
    os.makedirs('models', exist_ok=True)
    
    model_path = f'models/{component_name}_autoencoder.h5'
    scaler_path = f'models/{component_name}_scaler.pkl'
    pca_path = f'models/{component_name}_pca.pkl'
    threshold_path = f'models/{component_name}_threshold.pkl'
    stats_path = f'models/{component_name}_stats.pkl'
    
    autoencoder.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    joblib.dump(threshold, threshold_path)
    
    # Save component-specific stats
    stats = {
        'sensors': sensors,
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist(),
        'input_dim': input_dim,
        'encoding_dim': encoding_dim
    }
    joblib.dump(stats, stats_path)
    
    print(f"\nSaved artifacts:")
    print(f"  Model: {model_path}")
    print(f"  Scaler: {scaler_path}")
    print(f"  PCA: {pca_path}")
    print(f"  Threshold: {threshold_path}")
    print(f"  Stats: {stats_path}")
    
    return {
        'model': autoencoder,
        'scaler': scaler,
        'pca': pca,
        'threshold': threshold,
        'stats': stats
    }

def train_all_models():
    """Train autoencoders for all components."""
    
    print("="*60)
    print("COMPONENT-SPECIFIC AUTOENCODER TRAINING")
    print("="*60)
    print("\nThis script trains SEPARATE models for each component.")
    print("Each model only sees sensors for its specific component.")
    print("Models are NEVER cross-trained or merged.")
    
    results = {}
    
    for component in COMPONENT_CONFIG.keys():
        try:
            results[component] = train_component_model(component)
        except Exception as e:
            print(f"ERROR training {component}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nComponent models saved in 'models/' directory:")
    for component in results.keys():
        print(f"  ✅ {component}_autoencoder.h5")
    
    print("\nThresholds:")
    for component, result in results.items():
        print(f"  {component}: {result['threshold']:.6f}")

if __name__ == '__main__':
    train_all_models()
