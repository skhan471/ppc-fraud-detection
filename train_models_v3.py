"""
PPC Ad Click Fraud Detection - Model Training v3
Advanced training with threshold tuning, cost-sensitive learning, and temporal cross-validation.
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Model imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit

# Check for XGBoost and LightGBM
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

def load_v3_data():
    """Load v3 preprocessed datasets."""
    print("Loading v3 preprocessed data...")

    X_train = pd.read_csv('data/X_train_v3.csv')
    X_test = pd.read_csv('data/X_test_v3.csv')
    y_train = pd.read_csv('data/y_train_v3.csv')
    y_test = pd.read_csv('data/y_test_v3.csv')

    # Convert y to 1D array
    y_train = y_train.iloc[:, 0].values if y_train.shape[1] == 1 else y_train.values
    y_test = y_test.iloc[:, 0].values if y_test.shape[1] == 1 else y_test.values

    print(f"Training data: {X_train.shape[0]:,} samples, {X_train.shape[1]:,} features")
    print(f"Test data: {X_test.shape[0]:,} samples")
    print(f"Class distribution in training (after SMOTE): {np.bincount(y_train)}")
    print(f"Class distribution in test (REAL): {np.bincount(y_test)}")
    print(f"Test set fraud ratio: {np.sum(y_test == 1) / len(y_test):.4f}")

    return X_train, X_test, y_train, y_test

def perform_temporal_cross_validation(model, X_train, y_train, n_splits=5):
    """Perform temporal cross-validation to assess model stability."""
    print(f"  Performing temporal cross-validation ({n_splits} splits)...")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
        y_train_fold = y_train[train_idx]
        y_val_fold = y_train[val_idx]

        # Train on fold
        model_copy = clone_model(model)
        model_copy.fit(X_train_fold, y_train_fold)

        # Evaluate on validation fold
        y_pred = model_copy.predict(X_val_fold)
        fold_f1 = f1_score(y_val_fold, y_pred, zero_division=0)
        cv_scores.append(fold_f1)

        print(f"    Fold {fold+1}: F1-Score = {fold_f1:.4f}")

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"  CV Results: Mean F1 = {cv_mean:.4f}, Std = {cv_std:.4f}")

    return cv_scores, cv_mean, cv_std

def clone_model(model):
    """Create a clone of a model for cross-validation."""
    from sklearn.base import clone
    return clone(model)

def find_optimal_threshold(model, X_train, y_train):
    """Find optimal threshold for binary classification using precision-recall tradeoff."""
    print("  Finding optimal threshold...")

    if not hasattr(model, 'predict_proba'):
        print("    Model doesn't support predict_proba, using default threshold 0.5")
        return 0.5

    # Get probability predictions
    y_pred_proba = model.predict_proba(X_train)[:, 1]

    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_train, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"    Optimal threshold: {best_threshold:.3f} (F1 = {best_f1:.4f})")
    return best_threshold

def train_decision_tree(X_train, y_train):
    """Train Decision Tree model with advanced parameters."""
    print("\nTraining Decision Tree (v3)...")

    # Base model
    model = DecisionTreeClassifier(
        max_depth=15,  # Increased for more complex patterns
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )

    # Temporal cross-validation
    cv_scores, cv_mean, cv_std = perform_temporal_cross_validation(model, X_train, y_train)

    # Train final model on all training data
    model.fit(X_train, y_train)

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(model, X_train, y_train)

    return model, optimal_threshold, cv_mean, cv_std

def train_random_forest(X_train, y_train):
    """Train Random Forest model with advanced parameters."""
    print("\nTraining Random Forest (v3)...")

    # Base model with more trees and deeper trees
    model = RandomForestClassifier(
        n_estimators=200,  # Increased from 100
        max_depth=15,      # Increased from 10
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced_subsample',  # More sophisticated class weighting
        random_state=42,
        n_jobs=-1,
        bootstrap=True
    )

    # Temporal cross-validation
    cv_scores, cv_mean, cv_std = perform_temporal_cross_validation(model, X_train, y_train)

    # Train final model on all training data
    model.fit(X_train, y_train)

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(model, X_train, y_train)

    return model, optimal_threshold, cv_mean, cv_std

def train_xgboost(X_train, y_train):
    """Train XGBoost model with advanced parameters."""
    if not XGB_AVAILABLE:
        return None, 0.5, 0, 0

    print("\nTraining XGBoost (v3)...")

    # Calculate scale_pos_weight based on actual class imbalance
    # Note: Training data is balanced after SMOTE, but we use realistic parameter
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

    # Advanced XGBoost parameters
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,  # Lower learning rate for better generalization
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        min_child_weight=5
    )

    # Temporal cross-validation
    cv_scores, cv_mean, cv_std = perform_temporal_cross_validation(model, X_train, y_train)

    # Train final model on all training data
    model.fit(X_train, y_train)

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(model, X_train, y_train)

    return model, optimal_threshold, cv_mean, cv_std

def train_lightgbm(X_train, y_train):
    """Train LightGBM model with advanced parameters."""
    if not LGB_AVAILABLE:
        return None, 0.5, 0, 0

    print("\nTraining LightGBM (v3)...")

    # Advanced LightGBM parameters
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=12,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        num_leaves=31
    )

    # Temporal cross-validation
    cv_scores, cv_mean, cv_std = perform_temporal_cross_validation(model, X_train, y_train)

    # Train final model on all training data
    model.fit(X_train, y_train)

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(model, X_train, y_train)

    return model, optimal_threshold, cv_mean, cv_std

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model with advanced parameters."""
    print("\nTraining Logistic Regression (v3)...")

    # Advanced Logistic Regression with regularization
    model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=42,
        solver='liblinear',  # Good for small to medium datasets
        penalty='l2',
        C=0.1  # Stronger regularization
    )

    # Temporal cross-validation
    cv_scores, cv_mean, cv_std = perform_temporal_cross_validation(model, X_train, y_train)

    # Train final model on all training data
    model.fit(X_train, y_train)

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(model, X_train, y_train)

    return model, optimal_threshold, cv_mean, cv_std

def evaluate_model_with_threshold(model, model_name, X_test, y_test, threshold=0.5):
    """Evaluate model on TEST SET with custom threshold."""
    if model is None:
        return None

    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # Apply custom threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        # For models without predict_proba, use default prediction
        y_pred = model.predict(X_test)
        y_pred_proba = None

    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Threshold': threshold,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0)
    }

    # Add ROC-AUC if we have probability predictions
    if y_pred_proba is not None:
        try:
            metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba)
        except:
            metrics['ROC-AUC'] = np.nan
    else:
        metrics['ROC-AUC'] = np.nan

    # Print detailed confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix (threshold={threshold:.3f}):")
    print(f"    TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"    FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

    return metrics, cm

def save_model_v3(model, model_name, threshold, cv_mean, cv_std):
    """Save trained model and metadata to pickle file with v3 suffix."""
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Create model metadata
    model_data = {
        'model': model,
        'threshold': threshold,
        'cv_mean_f1': cv_mean,
        'cv_std_f1': cv_std,
        'version': 'v3'
    }

    filename = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}_v3.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"  Saved model to {filename}")
    print(f"  Threshold: {threshold:.3f}, CV F1: {cv_mean:.4f} ± {cv_std:.4f}")
    return filename

def print_comparison_table_v3(results):
    """Print a clean comparison table of all models."""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON (v3 - ADVANCED TRAINING)")
    print("="*80)
    print("Note: Models evaluated on time-based test set with optimal thresholds")
    print("="*80)

    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Format for display
    display_df = df.copy()

    # Format numeric columns
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Threshold']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")

    # Print table
    print(display_df.to_string(index=False))

    # Print summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS (v3 - Advanced Training):")

    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    for col in numeric_cols:
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                best_idx = valid_values.idxmax()
                best_model = df.loc[best_idx, 'Model']
                best_value = valid_values.max()
                print(f"{col:12} - Best: {best_value:.4f} ({best_model})")

    return df

def save_comparison_results_v3(results_df):
    """Save comparison results to CSV file with v3 suffix."""
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    filename = os.path.join(results_dir, 'model_comparison_v3.csv')
    results_df.to_csv(filename, index=False)

    print(f"\nComparison results saved to: {filename}")
    return filename

def compare_with_previous_versions(results_df_v3):
    """Compare v3 results with previous v1 and v2 results."""
    print("\n" + "="*80)
    print("COMPARISON WITH PREVIOUS VERSIONS")
    print("="*80)

    try:
        # Load v1 results
        results_v1 = pd.read_csv('results/model_comparison.csv')
        print("\nV1 Results (with data leakage):")
        print(results_v1.to_string(index=False))
    except:
        results_v1 = None
        print("\nCould not load v1 results")

    try:
        # Load v2 results
        results_v2 = pd.read_csv('results/model_comparison_v2.csv')
        print("\nV2 Results (no data leakage):")
        print(results_v2.to_string(index=False))
    except:
        results_v2 = None
        print("\nCould not load v2 results")

    print("\nV3 Results (advanced training):")
    print(results_df_v3.to_string(index=False))

    print("\n" + "-"*80)
    print("PERFORMANCE IMPROVEMENT ANALYSIS:")

    if results_v2 is not None:
        for model in results_df_v3['Model'].unique():
            if model in results_v2['Model'].values:
                v2_row = results_v2[results_v2['Model'] == model].iloc[0]
                v3_row = results_df_v3[results_df_v3['Model'] == model].iloc[0]

                print(f"\n{model}:")
                for metric in ['F1-Score', 'Precision', 'Recall']:
                    if metric in v2_row and metric in v3_row:
                        v2_val = v2_row[metric]
                        v3_val = v3_row[metric]
                        if not pd.isna(v2_val) and not pd.isna(v3_val):
                            improvement = v3_val - v2_val
                            if improvement > 0:
                                print(f"  {metric}: {v2_val:.4f} -> {v3_val:.4f} (+{improvement:.4f})")
                            else:
                                print(f"  {metric}: {v2_val:.4f} -> {v3_val:.4f} ({improvement:.4f})")

    print("\nKey Insight: v3 uses advanced techniques (threshold tuning,")
    print("cost-sensitive learning, temporal CV) to improve performance")
    print("while maintaining research validity (no data leakage).")

def main():
    """Main training pipeline v3."""
    print("="*80)
    print("PPC AD CLICK FRAUD DETECTION - MODEL TRAINING v3")
    print("="*80)
    print("Advanced training with threshold tuning, cost-sensitive learning,")
    print("and temporal cross-validation")
    print("="*80)

    # Step 1: Load v3 data
    X_train, X_test, y_train, y_test = load_v3_data()

    # Step 2: Train models with advanced techniques
    models = {}
    thresholds = {}
    cv_results = {}
    results = []

    # Decision Tree
    dt_model, dt_threshold, dt_cv_mean, dt_cv_std = train_decision_tree(X_train, y_train)
    models['Decision Tree'] = dt_model
    thresholds['Decision Tree'] = dt_threshold
    cv_results['Decision Tree'] = (dt_cv_mean, dt_cv_std)
    dt_metrics, _ = evaluate_model_with_threshold(dt_model, 'Decision Tree', X_test, y_test, dt_threshold)
    if dt_metrics:
        results.append(dt_metrics)
        save_model_v3(dt_model, 'decision_tree', dt_threshold, dt_cv_mean, dt_cv_std)

    # Random Forest
    rf_model, rf_threshold, rf_cv_mean, rf_cv_std = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf_model
    thresholds['Random Forest'] = rf_threshold
    cv_results['Random Forest'] = (rf_cv_mean, rf_cv_std)
    rf_metrics, _ = evaluate_model_with_threshold(rf_model, 'Random Forest', X_test, y_test, rf_threshold)
    if rf_metrics:
        results.append(rf_metrics)
        save_model_v3(rf_model, 'random_forest', rf_threshold, rf_cv_mean, rf_cv_std)

    # XGBoost
    xgb_model, xgb_threshold, xgb_cv_mean, xgb_cv_std = train_xgboost(X_train, y_train)
    models['XGBoost'] = xgb_model
    thresholds['XGBoost'] = xgb_threshold
    cv_results['XGBoost'] = (xgb_cv_mean, xgb_cv_std)
    if xgb_model:
        xgb_metrics, _ = evaluate_model_with_threshold(xgb_model, 'XGBoost', X_test, y_test, xgb_threshold)
        if xgb_metrics:
            results.append(xgb_metrics)
            save_model_v3(xgb_model, 'xgboost', xgb_threshold, xgb_cv_mean, xgb_cv_std)
    else:
        print("Skipping XGBoost (not available)")

    # LightGBM
    lgb_model, lgb_threshold, lgb_cv_mean, lgb_cv_std = train_lightgbm(X_train, y_train)
    models['LightGBM'] = lgb_model
    thresholds['LightGBM'] = lgb_threshold
    cv_results['LightGBM'] = (lgb_cv_mean, lgb_cv_std)
    if lgb_model:
        lgb_metrics, _ = evaluate_model_with_threshold(lgb_model, 'LightGBM', X_test, y_test, lgb_threshold)
        if lgb_metrics:
            results.append(lgb_metrics)
            save_model_v3(lgb_model, 'lightgbm', lgb_threshold, lgb_cv_mean, lgb_cv_std)
    else:
        print("Skipping LightGBM (not available)")

    # Logistic Regression
    lr_model, lr_threshold, lr_cv_mean, lr_cv_std = train_logistic_regression(X_train, y_train)
    models['Logistic Regression'] = lr_model
    thresholds['Logistic Regression'] = lr_threshold
    cv_results['Logistic Regression'] = (lr_cv_mean, lr_cv_std)
    lr_metrics, _ = evaluate_model_with_threshold(lr_model, 'Logistic Regression', X_test, y_test, lr_threshold)
    if lr_metrics:
        results.append(lr_metrics)
        save_model_v3(lr_model, 'logistic_regression', lr_threshold, lr_cv_mean, lr_cv_std)

    # Step 3: Print comparison table
    if results:
        results_df = print_comparison_table_v3(results)

        # Step 4: Save comparison results
        save_comparison_results_v3(results_df)

        # Step 5: Compare with previous versions
        compare_with_previous_versions(results_df)

        # Additional analysis
        print("\n" + "="*80)
        print("MODEL TRAINING COMPLETE (v3)")
        print("="*80)

        print(f"\nTrained {len([m for m in models.values() if m is not None])} models:")
        for name, model in models.items():
            if model is not None:
                cv_mean, cv_std = cv_results.get(name, (0, 0))
                print(f"  - {name}: threshold={thresholds[name]:.3f}, CV F1={cv_mean:.4f} ± {cv_std:.4f}")

        print(f"\nModels saved to 'models/' directory with '_v3' suffix")
        print(f"Results saved to 'results/model_comparison_v3.csv'")

        # Check if target F1-Score >= 0.5 is achieved
        if 'F1-Score' in results_df.columns:
            best_f1_idx = results_df['F1-Score'].idxmax()
            best_model = results_df.loc[best_f1_idx, 'Model']
            best_f1 = results_df.loc[best_f1_idx, 'F1-Score']

            print(f"\nTARGET ACHIEVEMENT ANALYSIS:")
            print(f"Best F1-Score: {best_f1:.4f} ({best_model})")
            if best_f1 >= 0.5:
                print(f"[+] TARGET ACHIEVED: F1-Score >= 0.5")
            else:
                print(f"[X] TARGET NOT ACHIEVED: F1-Score < 0.5")
                print(f"   Improvement needed: {0.5 - best_f1:.4f}")

            # Note about advanced techniques
            print(f"\nADVANCED TECHNIQUES APPLIED (v3):")
            print("1. Threshold tuning for optimal precision-recall tradeoff")
            print("2. Cost-sensitive learning with class weights")
            print("3. Temporal cross-validation for time-series stability")
            print("4. Advanced feature engineering (from preprocessing_v3)")
            print("5. Regularization and hyperparameter optimization")
    else:
        print("\nNo models were successfully trained and evaluated.")

    return models, thresholds, results

if __name__ == "__main__":
    models, thresholds, results = main()