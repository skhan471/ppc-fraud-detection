"""
PPC Ad Click Fraud Detection - Preprocessing Pipeline v2
Fixes data leakage with proper time-based split and feature engineering.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For SMOTE
from imblearn.over_sampling import SMOTE

def load_and_prepare_data():
    """Load raw data and convert click_time to datetime."""
    print("Loading raw dataset...")

    # Load the dataset
    df = pd.read_csv('data/train_sample.csv')

    # Convert click_time to datetime
    df['click_time'] = pd.to_datetime(df['click_time'])

    # Sort by click_time (IMPORTANT for time-based split)
    df = df.sort_values('click_time').reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    print(f"Time range: {df['click_time'].min()} to {df['click_time'].max()}")

    return df

def time_based_split(df, train_ratio=0.8):
    """Split data by time (first train_ratio% for training, rest for testing)."""
    print(f"\nPerforming time-based split ({train_ratio*100:.0f}% train, {(1-train_ratio)*100:.0f}% test)...")

    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Training set: {len(train_df):,} samples ({train_ratio*100:.0f}%)")
    print(f"Test set: {len(test_df):,} samples ({(1-train_ratio)*100:.0f}%)")
    print(f"Training time range: {train_df['click_time'].min()} to {train_df['click_time'].max()}")
    print(f"Test time range: {test_df['click_time'].min()} to {test_df['click_time'].max()}")

    return train_df, test_df

def create_time_features(df):
    """Create time-based features from click_time."""
    print("Creating time-based features...")

    df['click_hour'] = df['click_time'].dt.hour
    df['click_day'] = df['click_time'].dt.day
    df['click_minute'] = df['click_time'].dt.minute
    df['click_second'] = df['click_time'].dt.second
    df['click_dayofmonth'] = df['click_time'].dt.day

    return df

def create_frequency_features_train(train_df):
    """Create frequency encoding features using ONLY training data."""
    print("\nCreating frequency encoding features (training data only)...")

    # Store mappings for applying to test data
    frequency_mappings = {}

    # Individual frequency features
    for col in ['ip', 'app', 'device', 'channel']:
        freq = train_df[col].value_counts()
        train_df[f'{col}_frequency'] = train_df[col].map(freq)
        frequency_mappings[f'{col}_frequency'] = freq.to_dict()

    # Combined frequency features
    train_df['ip_app'] = train_df['ip'].astype(str) + '_' + train_df['app'].astype(str)
    train_df['ip_device'] = train_df['ip'].astype(str) + '_' + train_df['device'].astype(str)

    ip_app_freq = train_df['ip_app'].value_counts()
    ip_device_freq = train_df['ip_device'].value_counts()

    train_df['ip_app_frequency'] = train_df['ip_app'].map(ip_app_freq)
    train_df['ip_device_frequency'] = train_df['ip_device'].map(ip_device_freq)

    frequency_mappings['ip_app_frequency'] = ip_app_freq.to_dict()
    frequency_mappings['ip_device_frequency'] = ip_device_freq.to_dict()

    # Drop temporary columns
    train_df = train_df.drop(['ip_app', 'ip_device'], axis=1)

    return train_df, frequency_mappings

def apply_frequency_features_test(test_df, frequency_mappings):
    """Apply frequency encoding to test data using training mappings."""
    print("Applying frequency features to test data...")

    # Individual frequency features
    for col in ['ip', 'app', 'device', 'channel']:
        mapping = frequency_mappings.get(f'{col}_frequency', {})
        test_df[f'{col}_frequency'] = test_df[col].map(mapping).fillna(0).astype(int)

    # Combined frequency features
    test_df['ip_app'] = test_df['ip'].astype(str) + '_' + test_df['app'].astype(str)
    test_df['ip_device'] = test_df['ip'].astype(str) + '_' + test_df['device'].astype(str)

    ip_app_mapping = frequency_mappings.get('ip_app_frequency', {})
    ip_device_mapping = frequency_mappings.get('ip_device_frequency', {})

    test_df['ip_app_frequency'] = test_df['ip_app'].map(ip_app_mapping).fillna(0).astype(int)
    test_df['ip_device_frequency'] = test_df['ip_device'].map(ip_device_mapping).fillna(0).astype(int)

    # Drop temporary columns
    test_df = test_df.drop(['ip_app', 'ip_device'], axis=1)

    return test_df

def prepare_features_and_target(train_df, test_df):
    """Prepare X and y for training and testing."""
    print("\nPreparing features and target...")

    # Drop time columns
    features_to_drop = ['click_time', 'attributed_time']

    # Prepare training data
    X_train = train_df.drop(features_to_drop + ['is_attributed'], axis=1)
    y_train = train_df['is_attributed']

    # Prepare test data
    X_test = test_df.drop(features_to_drop + ['is_attributed'], axis=1)
    y_test = test_df['is_attributed']

    print(f"Training features: {X_train.shape}")
    print(f"Test features: {X_test.shape}")

    return X_train, y_train, X_test, y_test

def apply_smote(X_train, y_train):
    """Apply SMOTE to training data only."""
    print("\nApplying SMOTE to training data...")

    print(f"Class distribution BEFORE SMOTE:")
    print(f"  Class 0 (Legitimate): {np.sum(y_train == 0):,}")
    print(f"  Class 1 (Fraud): {np.sum(y_train == 1):,}")
    print(f"  Ratio: {np.sum(y_train == 0) / np.sum(y_train == 1):.2f}:1")

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"\nClass distribution AFTER SMOTE:")
    print(f"  Class 0 (Legitimate): {np.sum(y_train_resampled == 0):,}")
    print(f"  Class 1 (Fraud): {np.sum(y_train_resampled == 1):,}")
    print(f"  Ratio: 1:1 (balanced)")

    return X_train_resampled, y_train_resampled

def save_datasets(X_train, y_train, X_test, y_test, suffix='_v2'):
    """Save processed datasets."""
    print(f"\nSaving datasets with suffix '{suffix}'...")

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Save datasets
    X_train.to_csv(f'data/X_train{suffix}.csv', index=False)
    X_test.to_csv(f'data/X_test{suffix}.csv', index=False)
    y_train.to_csv(f'data/y_train{suffix}.csv', index=False)
    y_test.to_csv(f'data/y_test{suffix}.csv', index=False)

    # Save feature names
    with open(f'data/feature_names{suffix}.txt', 'w') as f:
        for feature in X_train.columns:
            f.write(f"{feature}\n")

    print(f"Saved datasets:")
    print(f"  - data/X_train{suffix}.csv ({X_train.shape})")
    print(f"  - data/X_test{suffix}.csv ({X_test.shape})")
    print(f"  - data/y_train{suffix}.csv ({len(y_train)})")
    print(f"  - data/y_test{suffix}.csv ({len(y_test)})")
    print(f"  - data/feature_names{suffix}.txt ({len(X_train.columns)} features)")

def main():
    """Main preprocessing pipeline."""
    print("="*80)
    print("PPC AD CLICK FRAUD DETECTION - PREPROCESSING PIPELINE v2")
    print("="*80)
    print("Fixing data leakage with time-based split and proper feature engineering")
    print("="*80)

    # Step 1: Load and prepare data
    df = load_and_prepare_data()

    # Step 2: Time-based split
    train_df, test_df = time_based_split(df, train_ratio=0.8)

    # Step 3: Create time features on both sets
    train_df = create_time_features(train_df)
    test_df = create_time_features(test_df)

    # Step 4: Create frequency features using ONLY training data
    train_df, frequency_mappings = create_frequency_features_train(train_df)

    # Step 5: Apply frequency features to test data using training mappings
    test_df = apply_frequency_features_test(test_df, frequency_mappings)

    # Step 6: Prepare features and target
    X_train, y_train, X_test, y_test = prepare_features_and_target(train_df, test_df)

    # Step 7: Apply SMOTE to training data only
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    # Step 8: Save datasets
    save_datasets(X_train_resampled, y_train_resampled, X_test, y_test, suffix='_v2')

    # Final summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\nDataset Summary:")
    print(f"  Original dataset: {len(df):,} samples")
    print(f"  Training set (after SMOTE): {len(X_train_resampled):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    print(f"  Number of features: {X_train_resampled.shape[1]}")
    print(f"\nKey methodological improvements:")
    print(f"  1. Time-based split (no data leakage)")
    print(f"  2. Feature engineering using training data only")
    print(f"  3. SMOTE applied only to training data")
    print(f"  4. Realistic evaluation setup")

    return X_train_resampled, y_train_resampled, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = main()