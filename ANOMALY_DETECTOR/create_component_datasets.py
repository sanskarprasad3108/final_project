"""
Create Component-Specific Datasets for Multi-Interface Anomaly Detection
=========================================================================
Splits the main dataset into component-specific datasets with aligned timestamps.
Each component dataset contains ONLY sensors relevant to that component.

Component → Sensor Mapping:
- ENGINE: engine_temp_C, oil_pressure_bar, fuel_rate_lph
- HYDRAULIC: hydraulic_pressure_bar, load_tons
- WHEELS: vibration_mm_s, brake_temp_C, speed_kmph
- CHASSIS: vibration_mm_s, load_tons (structural stress indicators)
"""

import pandas as pd
import numpy as np
import os

# Define component-to-sensor mapping
COMPONENT_SENSORS = {
    'engine': ['engine_temp_C', 'oil_pressure_bar', 'fuel_rate_lph'],
    'hydraulic': ['hydraulic_pressure_bar', 'load_tons'],
    'wheels': ['vibration_mm_s', 'brake_temp_C', 'speed_kmph'],
    'chassis': ['vibration_mm_s', 'load_tons']
}

def create_component_datasets():
    """Split main dataset into component-specific datasets with aligned timestamps."""
    
    # Load main dataset
    print("Loading main dataset...")
    df = pd.read_csv('dataset.csv.csv')
    print(f"Main dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Ensure datasets directory exists
    os.makedirs('datasets', exist_ok=True)
    
    # Create each component dataset
    for component, sensors in COMPONENT_SENSORS.items():
        print(f"\n{'='*50}")
        print(f"Creating {component.upper()} dataset")
        print(f"{'='*50}")
        
        # Columns to include: timestamp + sensors + label
        columns = ['timestamp'] + sensors + ['label']
        
        # Check if all columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            print(f"WARNING: Missing columns for {component}: {missing_cols}")
            # Try alternative column names
            continue
        
        # Extract component-specific data
        component_df = df[columns].copy()
        
        # Save to CSV
        output_path = f'datasets/{component}.csv'
        component_df.to_csv(output_path, index=False)
        
        print(f"Sensors: {sensors}")
        print(f"Shape: {component_df.shape}")
        print(f"Normal samples: {len(component_df[component_df['label'] == 0])}")
        print(f"Anomaly samples: {len(component_df[component_df['label'] == 1])}")
        print(f"Saved to: {output_path}")
        
        # Print sample statistics
        print("\nSensor Statistics:")
        for sensor in sensors:
            print(f"  {sensor}: mean={component_df[sensor].mean():.2f}, std={component_df[sensor].std():.2f}")
    
    print(f"\n{'='*50}")
    print("ALL COMPONENT DATASETS CREATED SUCCESSFULLY")
    print(f"{'='*50}")
    print("\nDatasets saved in 'datasets/' directory:")
    print("  - engine.csv")
    print("  - hydraulic.csv")
    print("  - wheels.csv")
    print("  - chassis.csv")

if __name__ == '__main__':
    create_component_datasets()
