"""
PPC Ad Click Fraud Detection - Model Training Script
Trains and evaluates 5 different models for fraud detection.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

def load_data():
    """Load preprocessed training and test data."""
    print("Loading preprocessed data...")

    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv')
    y_test = pd.read_csv('data/y_test.csv')

    # Convert y to 1D array
    y_train = y_train['is_attributed'].values
    y_test = y_test['is_attributed'].values

    print(f"Training data: {X_train.shape[0]:,} samples, {X_train.shape[1]:,} features")
    print(f"Test data: {X_test.shape[0]:,} samples")
    print(f"Class distribution in training: {np.bincount(y_train)}")
    print(f"Class distribution in test: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train):
    """Train Decision Tree model."""
    print("\nTraining Decision Tree...")
    model = DecisionTreeClassifier(
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Train XGBoost model."""
    if not XGB_AVAILABLE:
        return None

    print("Training XGBoost...")

    # Calculate scale_pos_weight for class imbalance
    # Since we used SMOTE, data is balanced, but we keep the parameter
    model = xgb.XGBClassifier(
        n_estimators=100,
        scale_pos_weight=1,  # Balanced after SMOTE
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    """Train LightGBM model."""
    if not LGB_AVAILABLE:
        return None

    print("Training LightGBM...")

    # LightGBM needs class weights as dictionary
    class_counts = np.bincount(y_train)
    class_weights = {0: 1.0, 1: class_counts[0] / class_counts[1]}

    model = lgb.LGBMClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        verbose=-1  # Suppress output
    )
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    print("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, model_name, X_test, y_test):
    """Evaluate model and return metrics."""
    if model is None:
        return None

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }

    # Add ROC-AUC if we have probability predictions
    if y_pred_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba)
    else:
        metrics['ROC-AUC'] = np.nan

    return metrics

def save_model(model, model_name):
    """Save trained model to pickle file."""
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    filename = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"  Saved model to {filename}")
    return filename

def print_comparison_table(results):
    """Print a clean comparison table of all models."""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)

    # Create DataFrame from results
    df = pd.DataFrame(results)

    # Format for display
    display_df = df.copy()

    # Format numeric columns
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")

    # Print table
    print(display_df.to_string(index=False))

    # Print summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS (excluding N/A values):")

    numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    for col in numeric_cols:
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                print(f"{col:12} - Best: {valid_values.max():.4f} ({df.loc[valid_values.idxmax(), 'Model']})")

    return df

def save_comparison_results(results_df):
    """Save comparison results to CSV file."""
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    filename = os.path.join(results_dir, 'model_comparison.csv')
    results_df.to_csv(filename, index=False)

    print(f"\nComparison results saved to: {filename}")
    return filename

def main():
    """Main training pipeline."""
    print("="*80)
    print("PPC AD CLICK FRAUD DETECTION - MODEL TRAINING")
    print("="*80)

    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_data()

    # Step 2: Train models
    models = {}
    results = []

    # Decision Tree
    dt_model = train_decision_tree(X_train, y_train)
    models['Decision Tree'] = dt_model
    dt_metrics = evaluate_model(dt_model, 'Decision Tree', X_test, y_test)
    if dt_metrics:
        results.append(dt_metrics)
        save_model(dt_model, 'decision_tree')

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf_model
    rf_metrics = evaluate_model(rf_model, 'Random Forest', X_test, y_test)
    if rf_metrics:
        results.append(rf_metrics)
        save_model(rf_model, 'random_forest')

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    models['XGBoost'] = xgb_model
    if xgb_model:
        xgb_metrics = evaluate_model(xgb_model, 'XGBoost', X_test, y_test)
        if xgb_metrics:
            results.append(xgb_metrics)
            save_model(xgb_model, 'xgboost')
    else:
        print("Skipping XGBoost (not available)")

    # LightGBM
    lgb_model = train_lightgbm(X_train, y_train)
    models['LightGBM'] = lgb_model
    if lgb_model:
        lgb_metrics = evaluate_model(lgb_model, 'LightGBM', X_test, y_test)
        if lgb_metrics:
            results.append(lgb_metrics)
            save_model(lgb_model, 'lightgbm')
    else:
        print("Skipping LightGBM (not available)")

    # Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    models['Logistic Regression'] = lr_model
    lr_metrics = evaluate_model(lr_model, 'Logistic Regression', X_test, y_test)
    if lr_metrics:
        results.append(lr_metrics)
        save_model(lr_model, 'logistic_regression')

    # Step 3: Print comparison table
    if results:
        results_df = print_comparison_table(results)

        # Step 4: Save comparison results
        save_comparison_results(results_df)

        # Additional analysis
        print("\n" + "="*80)
        print("MODEL TRAINING COMPLETE")
        print("="*80)

        print(f"\nTrained {len(models)} models:")
        for name, model in models.items():
            if model is not None:
                print(f"  - {name}")

        print(f"\nModels saved to 'models/' directory")
        print(f"Results saved to 'results/model_comparison.csv'")

        # Recommendation based on F1-Score (balanced metric)
        if 'F1-Score' in results_df.columns:
            best_f1_idx = results_df['F1-Score'].idxmax()
            best_model = results_df.loc[best_f1_idx, 'Model']
            best_f1 = results_df.loc[best_f1_idx, 'F1-Score']
            print(f"\nRECOMMENDATION: {best_model} has the highest F1-Score ({best_f1:.4f})")

            # Also check ROC-AUC if available
            if 'ROC-AUC' in results_df.columns:
                best_auc_idx = results_df['ROC-AUC'].idxmax()
                best_auc_model = results_df.loc[best_auc_idx, 'Model']
                best_auc = results_df.loc[best_auc_idx, 'ROC-AUC']
                print(f"           {best_auc_model} has the highest ROC-AUC ({best_auc:.4f})")
    else:
        print("\nNo models were successfully trained and evaluated.")

    return models, results

if __name__ == "__main__":
    models, results = main()