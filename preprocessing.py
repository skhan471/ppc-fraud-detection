"""
PPC Ad Click Fraud Detection - Preprocessing Pipeline
This script loads the dataset, performs feature engineering, handles class imbalance,
and creates train/test splits for modeling.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# For SMOTE and train/test split
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    """Load the dataset from CSV file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df

def drop_attributed_time(df):
    """Drop the attributed_time column (99.8% missing)."""
    if 'attributed_time' in df.columns:
        print(f"\nDropping 'attributed_time' column (99.8% missing values)...")
        df = df.drop('attributed_time', axis=1)
        print(f"Remaining columns: {list(df.columns)}")
    return df

def engineer_time_features(df):
    """Convert click_time to datetime and create time-based features."""
    print("\nEngineering time-based features from 'click_time'...")

    # Convert to datetime
    df['click_time'] = pd.to_datetime(df['click_time'])

    # Extract time components
    df['click_hour'] = df['click_time'].dt.hour
    df['click_day'] = df['click_time'].dt.dayofweek  # Monday=0, Sunday=6
    df['click_minute'] = df['click_time'].dt.minute

    # Additional time features that might be useful
    df['click_second'] = df['click_time'].dt.second
    df['click_dayofmonth'] = df['click_time'].dt.day

    print(f"Created time features: click_hour, click_day, click_minute, click_second, click_dayofmonth")

    # Drop original click_time column
    df = df.drop('click_time', axis=1)

    return df

def create_frequency_features(df):
    """Create frequency encoding features for categorical columns."""
    print("\nCreating frequency encoding features...")

    # Individual frequency features
    df['ip_frequency'] = df.groupby('ip')['ip'].transform('count')
    df['app_frequency'] = df.groupby('app')['app'].transform('count')
    df['device_frequency'] = df.groupby('device')['device'].transform('count')
    df['os_frequency'] = df.groupby('os')['os'].transform('count')
    df['channel_frequency'] = df.groupby('channel')['channel'].transform('count')

    # Combined frequency features
    df['ip_app_frequency'] = df.groupby(['ip', 'app']).transform('size')
    df['ip_device_frequency'] = df.groupby(['ip', 'device']).transform('size')
    df['ip_channel_frequency'] = df.groupby(['ip', 'channel']).transform('size')
    df['app_channel_frequency'] = df.groupby(['app', 'channel']).transform('size')

    print("Created frequency features:")
    print("  - Individual: ip_frequency, app_frequency, device_frequency, os_frequency, channel_frequency")
    print("  - Combined: ip_app_frequency, ip_device_frequency, ip_channel_frequency, app_channel_frequency")

    return df

def separate_features_target(df):
    """Separate features (X) and target (y)."""
    print("\nSeparating features and target...")

    # Target column
    target_col = 'is_attributed'

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    y = df[target_col]
    X = df.drop(target_col, axis=1)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts().to_string()}")

    return X, y

def apply_smote_balancing(X, y):
    """Apply SMOTE to handle class imbalance."""
    print("\nApplying SMOTE to handle class imbalance...")

    # Check class distribution before SMOTE
    class_counts_before = y.value_counts()
    print(f"Class distribution before SMOTE:")
    for cls, count in class_counts_before.items():
        print(f"  Class {cls}: {count:,} samples ({count/len(y)*100:.2f}%)")

    # Apply SMOTE
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Check class distribution after SMOTE
    class_counts_after = pd.Series(y_resampled).value_counts()
    print(f"\nClass distribution after SMOTE:")
    for cls, count in class_counts_after.items():
        print(f"  Class {cls}: {count:,} samples ({count/len(y_resampled)*100:.2f}%)")

    print(f"\nSMOTE results:")
    print(f"  Original dataset: {len(X):,} samples")
    print(f"  After SMOTE: {len(X_resampled):,} samples")
    print(f"  Increase: {len(X_resampled) - len(X):,} samples")

    return X_resampled, y_resampled

def split_train_test(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification."""
    print(f"\nSplitting data into train/test sets ({int((1-test_size)*100)}/{int(test_size*100)})...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Preserve class distribution
    )

    print(f"Train set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    # Check class distribution in splits
    print(f"\nTrain set class distribution:")
    train_dist = y_train.value_counts()
    for cls, count in train_dist.items():
        print(f"  Class {cls}: {count:,} samples ({count/len(y_train)*100:.2f}%)")

    print(f"\nTest set class distribution:")
    test_dist = y_test.value_counts()
    for cls, count in test_dist.items():
        print(f"  Class {cls}: {count:,} samples ({count/len(y_test)*100:.2f}%)")

    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test, feature_names, output_dir='data'):
    """Save processed data to CSV files."""
    print(f"\nSaving processed data to '{output_dir}' directory...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save data
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    print(f"Saved files:")
    print(f"  - {output_dir}/X_train.csv ({X_train.shape[0]:,} rows, {X_train.shape[1]:,} cols)")
    print(f"  - {output_dir}/X_test.csv ({X_test.shape[0]:,} rows, {X_test.shape[1]:,} cols)")
    print(f"  - {output_dir}/y_train.csv ({len(y_train):,} rows)")
    print(f"  - {output_dir}/y_test.csv ({len(y_test):,} rows)")
    print(f"  - {output_dir}/feature_names.txt ({len(feature_names):,} features)")

def main():
    """Main preprocessing pipeline."""
    print("="*80)
    print("PPC AD CLICK FRAUD DETECTION - PREPROCESSING PIPELINE")
    print("="*80)

    # File paths
    data_file = 'data/train_sample.csv'
    output_dir = 'data'

    # Step 1: Load data
    df = load_data(data_file)

    # Step 2: Drop attributed_time column
    df = drop_attributed_time(df)

    # Step 3: Engineer time features
    df = engineer_time_features(df)

    # Step 4: Create frequency features
    df = create_frequency_features(df)

    # Step 5: Separate features and target
    X, y = separate_features_target(df)

    # Step 6: Apply SMOTE for class imbalance
    X_balanced, y_balanced = apply_smote_balancing(X, y)

    # Step 7: Split into train/test sets
    X_train, X_test, y_train, y_test = split_train_test(X_balanced, y_balanced)

    # Step 8: Save processed data
    save_data(X_train, X_test, y_train, y_test, X_train.columns.tolist(), output_dir)

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)

    # Final summary
    print("\nFINAL DATASET SUMMARY:")
    print(f"Original dataset: 100,000 samples")
    print(f"After preprocessing: {len(X_balanced):,} samples (balanced)")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"\nFeature categories:")
    print(f"  - Original categorical: ip, app, device, os, channel")
    print(f"  - Time-based: click_hour, click_day, click_minute, click_second, click_dayofmonth")
    print(f"  - Frequency encoding: 9 frequency features")
    print(f"\nFiles saved to '{output_dir}/' directory")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()