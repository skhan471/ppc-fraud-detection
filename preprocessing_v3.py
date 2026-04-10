"""
PPC Ad Click Fraud Detection - Advanced Preprocessing Pipeline v3
Advanced feature engineering with strict no-leakage constraints.
Target: F1-score ≥ 0.5 for fraud class.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For SMOTE
from imblearn.over_sampling import SMOTE

# For entropy calculation
from scipy.stats import entropy

def load_and_prepare_data():
    """Load raw data and convert click_time to datetime."""
    print("Loading raw dataset...")

    # Load the dataset
    df = pd.read_csv('data/train_sample.csv')

    # Convert click_time to datetime
    df['click_time'] = pd.to_datetime(df['click_time'])

    # Sort by click_time (CRITICAL for time-based split and rolling features)
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

def create_time_since_last_click_train(train_df):
    """Create 'time since last click' feature for each IP using training data only."""
    print("Creating 'time since last click' feature (training data)...")

    # Sort by ip and click_time for proper diff calculation
    train_df = train_df.sort_values(['ip', 'click_time']).reset_index(drop=True)

    # Calculate time difference from previous click of same IP
    train_df['time_since_last_click'] = train_df.groupby('ip')['click_time'].diff().dt.total_seconds()

    # First click for each IP gets NaN - fill with large value (e.g., 1 day)
    train_df['time_since_last_click'] = train_df['time_since_last_click'].fillna(24*3600)

    # Sort back by click_time
    train_df = train_df.sort_values('click_time').reset_index(drop=True)

    return train_df

def apply_time_since_last_click_test(test_df):
    """Apply 'time since last click' feature to test data using same logic."""
    print("Applying 'time since last click' feature to test data...")

    # Sort by ip and click_time
    test_df = test_df.sort_values(['ip', 'click_time']).reset_index(drop=True)

    # Calculate time difference from previous click of same IP
    test_df['time_since_last_click'] = test_df.groupby('ip')['click_time'].diff().dt.total_seconds()

    # First click for each IP gets NaN - fill with large value (e.g., 1 day)
    test_df['time_since_last_click'] = test_df['time_since_last_click'].fillna(24*3600)

    # Sort back by click_time
    test_df = test_df.sort_values('click_time').reset_index(drop=True)

    return test_df

def create_rolling_window_features_train(train_df):
    """Create rolling window features for behavioral patterns (training data only)."""
    print("Creating rolling window features (training data)...")

    # Make a copy to avoid modifying original
    train_df = train_df.copy()

    # Sort by ip and click_time
    train_df = train_df.sort_values(['ip', 'click_time']).reset_index(drop=True)

    # Convert click_time to timestamp for rolling calculations
    train_df['click_timestamp'] = train_df['click_time'].astype('int64') // 10**9

    # Initialize new columns
    train_df['clicks_last_1min'] = 0
    train_df['clicks_last_5min'] = 0
    train_df['clicks_last_15min'] = 0

    # Calculate rolling counts for each IP
    for ip in train_df['ip'].unique():
        ip_mask = train_df['ip'] == ip
        ip_data = train_df[ip_mask].copy()

        if len(ip_data) > 1:
            # Get timestamps for this IP
            timestamps = ip_data['click_timestamp'].values

            # Calculate rolling counts
            clicks_1min = []
            clicks_5min = []
            clicks_15min = []

            for i, ts in enumerate(timestamps):
                # Count clicks in last 1 minute (60 seconds)
                count_1min = np.sum((timestamps[:i+1] >= ts - 60) & (timestamps[:i+1] <= ts))
                clicks_1min.append(count_1min - 1)  # Subtract current click

                # Count clicks in last 5 minutes (300 seconds)
                count_5min = np.sum((timestamps[:i+1] >= ts - 300) & (timestamps[:i+1] <= ts))
                clicks_5min.append(count_5min - 1)

                # Count clicks in last 15 minutes (900 seconds)
                count_15min = np.sum((timestamps[:i+1] >= ts - 900) & (timestamps[:i+1] <= ts))
                clicks_15min.append(count_15min - 1)

            # Assign back to dataframe
            train_df.loc[ip_mask, 'clicks_last_1min'] = clicks_1min
            train_df.loc[ip_mask, 'clicks_last_5min'] = clicks_5min
            train_df.loc[ip_mask, 'clicks_last_15min'] = clicks_15min

    # Drop temporary timestamp column
    train_df = train_df.drop('click_timestamp', axis=1)

    # Sort back by click_time
    train_df = train_df.sort_values('click_time').reset_index(drop=True)

    return train_df

def apply_rolling_window_features_test(test_df):
    """Apply rolling window features to test data using same logic."""
    print("Applying rolling window features to test data...")

    # Make a copy
    test_df = test_df.copy()

    # Sort by ip and click_time
    test_df = test_df.sort_values(['ip', 'click_time']).reset_index(drop=True)

    # Convert click_time to timestamp
    test_df['click_timestamp'] = test_df['click_time'].astype('int64') // 10**9

    # Initialize new columns
    test_df['clicks_last_1min'] = 0
    test_df['clicks_last_5min'] = 0
    test_df['clicks_last_15min'] = 0

    # Calculate rolling counts for each IP
    for ip in test_df['ip'].unique():
        ip_mask = test_df['ip'] == ip
        ip_data = test_df[ip_mask].copy()

        if len(ip_data) > 1:
            timestamps = ip_data['click_timestamp'].values

            clicks_1min = []
            clicks_5min = []
            clicks_15min = []

            for i, ts in enumerate(timestamps):
                count_1min = np.sum((timestamps[:i+1] >= ts - 60) & (timestamps[:i+1] <= ts))
                clicks_1min.append(count_1min - 1)

                count_5min = np.sum((timestamps[:i+1] >= ts - 300) & (timestamps[:i+1] <= ts))
                clicks_5min.append(count_5min - 1)

                count_15min = np.sum((timestamps[:i+1] >= ts - 900) & (timestamps[:i+1] <= ts))
                clicks_15min.append(count_15min - 1)

            test_df.loc[ip_mask, 'clicks_last_1min'] = clicks_1min
            test_df.loc[ip_mask, 'clicks_last_5min'] = clicks_5min
            test_df.loc[ip_mask, 'clicks_last_15min'] = clicks_15min

    # Drop temporary timestamp column
    test_df = test_df.drop('click_timestamp', axis=1)

    # Sort back by click_time
    test_df = test_df.sort_values('click_time').reset_index(drop=True)

    return test_df

def create_entropy_features_train(train_df):
    """Create entropy features for IP behavioral diversity (training data only)."""
    print("Creating entropy features for IP behavioral diversity (training data)...")

    # Store entropy mappings for test data application
    entropy_mappings = {}

    # Calculate entropy for each IP in training data
    ip_entropy_data = []

    for ip in train_df['ip'].unique():
        ip_data = train_df[train_df['ip'] == ip]

        if len(ip_data) > 1:
            # App usage entropy
            app_counts = ip_data['app'].value_counts(normalize=True)
            app_entropy = entropy(app_counts.values)

            # Device usage entropy
            device_counts = ip_data['device'].value_counts(normalize=True)
            device_entropy = entropy(device_counts.values)

            # Channel usage entropy
            channel_counts = ip_data['channel'].value_counts(normalize=True)
            channel_entropy = entropy(channel_counts.values)
        else:
            # Single record - entropy is 0
            app_entropy = 0
            device_entropy = 0
            channel_entropy = 0

        ip_entropy_data.append({
            'ip': ip,
            'app_entropy': app_entropy,
            'device_entropy': device_entropy,
            'channel_entropy': channel_entropy
        })

    # Create entropy dataframe
    entropy_df = pd.DataFrame(ip_entropy_data)

    # Store mappings for test data
    entropy_mappings['entropy'] = entropy_df.set_index('ip').to_dict('index')

    # Merge entropy features back to training data
    train_df = train_df.merge(entropy_df, on='ip', how='left')

    # Fill any NaN values (shouldn't happen but just in case)
    train_df[['app_entropy', 'device_entropy', 'channel_entropy']] = train_df[['app_entropy', 'device_entropy', 'channel_entropy']].fillna(0)

    return train_df, entropy_mappings

def apply_entropy_features_test(test_df, entropy_mappings):
    """Apply entropy features to test data using training mappings."""
    print("Applying entropy features to test data...")

    # Get entropy mappings
    entropy_dict = entropy_mappings.get('entropy', {})

    # Initialize entropy columns
    test_df['app_entropy'] = 0.0
    test_df['device_entropy'] = 0.0
    test_df['channel_entropy'] = 0.0

    # Apply entropy values for IPs seen in training
    for ip in test_df['ip'].unique():
        if ip in entropy_dict:
            test_df.loc[test_df['ip'] == ip, 'app_entropy'] = entropy_dict[ip]['app_entropy']
            test_df.loc[test_df['ip'] == ip, 'device_entropy'] = entropy_dict[ip]['device_entropy']
            test_df.loc[test_df['ip'] == ip, 'channel_entropy'] = entropy_dict[ip]['channel_entropy']
        # Unseen IPs remain 0 (already initialized)

    return test_df

def create_frequency_features_train(train_df):
    """Create frequency encoding features using ONLY training data."""
    print("Creating frequency encoding features (training data only)...")

    # Store mappings for applying to test data
    frequency_mappings = {}

    # Individual frequency features
    for col in ['ip', 'app', 'device', 'channel', 'os']:
        freq = train_df[col].value_counts()
        train_df[f'{col}_frequency'] = train_df[col].map(freq)
        frequency_mappings[f'{col}_frequency'] = freq.to_dict()

    # Combined frequency features
    train_df['ip_app'] = train_df['ip'].astype(str) + '_' + train_df['app'].astype(str)
    train_df['ip_device'] = train_df['ip'].astype(str) + '_' + train_df['device'].astype(str)
    train_df['ip_channel'] = train_df['ip'].astype(str) + '_' + train_df['channel'].astype(str)
    train_df['app_channel'] = train_df['app'].astype(str) + '_' + train_df['channel'].astype(str)

    ip_app_freq = train_df['ip_app'].value_counts()
    ip_device_freq = train_df['ip_device'].value_counts()
    ip_channel_freq = train_df['ip_channel'].value_counts()
    app_channel_freq = train_df['app_channel'].value_counts()

    train_df['ip_app_frequency'] = train_df['ip_app'].map(ip_app_freq)
    train_df['ip_device_frequency'] = train_df['ip_device'].map(ip_device_freq)
    train_df['ip_channel_frequency'] = train_df['ip_channel'].map(ip_channel_freq)
    train_df['app_channel_frequency'] = train_df['app_channel'].map(app_channel_freq)

    frequency_mappings['ip_app_frequency'] = ip_app_freq.to_dict()
    frequency_mappings['ip_device_frequency'] = ip_device_freq.to_dict()
    frequency_mappings['ip_channel_frequency'] = ip_channel_freq.to_dict()
    frequency_mappings['app_channel_frequency'] = app_channel_freq.to_dict()

    # Drop temporary columns
    train_df = train_df.drop(['ip_app', 'ip_device', 'ip_channel', 'app_channel'], axis=1)

    return train_df, frequency_mappings

def apply_frequency_features_test(test_df, frequency_mappings):
    """Apply frequency encoding to test data using training mappings."""
    print("Applying frequency features to test data...")

    # Individual frequency features
    for col in ['ip', 'app', 'device', 'channel', 'os']:
        mapping = frequency_mappings.get(f'{col}_frequency', {})
        test_df[f'{col}_frequency'] = test_df[col].map(mapping).fillna(0).astype(int)

    # Combined frequency features
    test_df['ip_app'] = test_df['ip'].astype(str) + '_' + test_df['app'].astype(str)
    test_df['ip_device'] = test_df['ip'].astype(str) + '_' + test_df['device'].astype(str)
    test_df['ip_channel'] = test_df['ip'].astype(str) + '_' + test_df['channel'].astype(str)
    test_df['app_channel'] = test_df['app'].astype(str) + '_' + test_df['channel'].astype(str)

    ip_app_mapping = frequency_mappings.get('ip_app_frequency', {})
    ip_device_mapping = frequency_mappings.get('ip_device_frequency', {})
    ip_channel_mapping = frequency_mappings.get('ip_channel_frequency', {})
    app_channel_mapping = frequency_mappings.get('app_channel_frequency', {})

    test_df['ip_app_frequency'] = test_df['ip_app'].map(ip_app_mapping).fillna(0).astype(int)
    test_df['ip_device_frequency'] = test_df['ip_device'].map(ip_device_mapping).fillna(0).astype(int)
    test_df['ip_channel_frequency'] = test_df['ip_channel'].map(ip_channel_mapping).fillna(0).astype(int)
    test_df['app_channel_frequency'] = test_df['app_channel'].map(app_channel_mapping).fillna(0).astype(int)

    # Drop temporary columns
    test_df = test_df.drop(['ip_app', 'ip_device', 'ip_channel', 'app_channel'], axis=1)

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

    print(f"Training features: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test

def apply_smote_to_training(X_train, y_train):
    """Apply SMOTE to training data only (no leakage to test)."""
    print("\nApplying SMOTE to training data...")

    original_shape = X_train.shape
    original_distribution = np.bincount(y_train)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"Original training shape: {original_shape}")
    print(f"After SMOTE training shape: {X_train_smote.shape}")
    print(f"Original class distribution: {original_distribution}")
    print(f"After SMOTE class distribution: {np.bincount(y_train_smote)}")

    return X_train_smote, y_train_smote

def save_datasets_v3(X_train, X_test, y_train, y_test):
    """Save v3 datasets to CSV files."""
    print("\nSaving v3 datasets...")

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Save training data
    X_train.to_csv('data/X_train_v3.csv', index=False)
    y_train_df = pd.DataFrame(y_train, columns=['is_attributed'])
    y_train_df.to_csv('data/y_train_v3.csv', index=False)

    # Save test data
    X_test.to_csv('data/X_test_v3.csv', index=False)
    y_test_df = pd.DataFrame(y_test, columns=['is_attributed'])
    y_test_df.to_csv('data/y_test_v3.csv', index=False)

    print("Saved datasets:")
    print(f"  - data/X_train_v3.csv ({X_train.shape[0]:,} samples, {X_train.shape[1]:,} features)")
    print(f"  - data/y_train_v3.csv ({len(y_train):,} samples)")
    print(f"  - data/X_test_v3.csv ({X_test.shape[0]:,} samples)")
    print(f"  - data/y_test_v3.csv ({len(y_test):,} samples)")

def main():
    """Main preprocessing pipeline v3."""
    print("="*80)
    print("PPC AD CLICK FRAUD DETECTION - ADVANCED PREPROCESSING PIPELINE v3")
    print("="*80)
    print("Advanced feature engineering with strict no-leakage constraints")
    print("Target: F1-score ≥ 0.5 for fraud class")
    print("="*80)

    # Step 1: Load and prepare data
    df = load_and_prepare_data()

    # Step 2: Time-based split
    train_df, test_df = time_based_split(df, train_ratio=0.8)

    # Step 3: Create time features
    train_df = create_time_features(train_df)
    test_df = create_time_features(test_df)

    # Step 4: Create time since last click features
    train_df = create_time_since_last_click_train(train_df)
    test_df = apply_time_since_last_click_test(test_df)

    # Step 5: Create rolling window features
    train_df = create_rolling_window_features_train(train_df)
    test_df = apply_rolling_window_features_test(test_df)

    # Step 6: Create entropy features
    train_df, entropy_mappings = create_entropy_features_train(train_df)
    test_df = apply_entropy_features_test(test_df, entropy_mappings)

    # Step 7: Create frequency encoding features
    train_df, frequency_mappings = create_frequency_features_train(train_df)
    test_df = apply_frequency_features_test(test_df, frequency_mappings)

    # Step 8: Prepare features and target
    X_train, X_test, y_train, y_test = prepare_features_and_target(train_df, test_df)

    # Step 9: Apply SMOTE to training data only
    X_train_smote, y_train_smote = apply_smote_to_training(X_train, y_train)

    # Step 10: Save datasets
    save_datasets_v3(X_train_smote, X_test, y_train_smote, y_test)

    # Final summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE (v3)")
    print("="*80)
    print(f"\nAdvanced features created:")
    print(f"  1. Time since last click (seconds)")
    print(f"  2. Rolling window clicks (1min, 5min, 15min)")
    print(f"  3. Entropy features (app, device, channel diversity)")
    print(f"  4. Frequency encoding (individual and combined)")
    print(f"  5. Time-based features (hour, day, minute, second)")
    print(f"\nTotal features: {X_train_smote.shape[1]}")
    print(f"Training samples (after SMOTE): {X_train_smote.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    print(f"Training class distribution: {np.bincount(y_train_smote)}")
    print(f"Test class distribution (REAL): {np.bincount(y_test)}")
    print(f"\nDatasets saved to 'data/' directory with '_v3' suffix")

if __name__ == "__main__":
    main()