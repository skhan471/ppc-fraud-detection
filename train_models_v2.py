"""
PPC Ad Click Fraud Detection - Model Training v2
Trains models on properly preprocessed data with no data leakage.
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

def load_v2_data():
    """Load v2 preprocessed datasets."""
    print("Loading v2 preprocessed data...")

    X_train = pd.read_csv('data/X_train_v2.csv')
    X_test = pd.read_csv('data/X_test_v2.csv')
    y_train = pd.read_csv('data/y_train_v2.csv')
    y_test = pd.read_csv('data/y_test_v2.csv')

    # Convert y to 1D array
    y_train = y_train.iloc[:, 0].values if y_train.shape[1] == 1 else y_train.values
    y_test = y_test.iloc[:, 0].values if y_test.shape[1] == 1 else y_test.values

    print(f"Training data: {X_train.shape[0]:,} samples, {X_train.shape[1]:,} features")
    print(f"Test data: {X_test.shape[0]:,} samples")
    print(f"Class distribution in training (after SMOTE): {np.bincount(y_train)}")
    print(f"Class distribution in test (REAL): {np.bincount(y_test)}")
    print(f"Test set fraud ratio: {np.sum(y_test == 1) / len(y_test):.4f}")

    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train):
    """Train Decision Tree model with realistic parameters."""
    print("\nTraining Decision Tree...")
    model = DecisionTreeClassifier(
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model with realistic parameters."""
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Train XGBoost model with realistic parameters."""
    if not XGB_AVAILABLE:
        return None

    print("Training XGBoost...")

    # Calculate scale_pos_weight for class imbalance
    # Note: Training data is balanced after SMOTE, but we keep realistic parameter
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    """Train LightGBM model with realistic parameters."""
    if not LGB_AVAILABLE:
        return None

    print("Training LightGBM...")

    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        verbose=-1
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
    """Evaluate model on TEST SET ONLY."""
    if model is None:
        return None

    # Make predictions
    y_pred = model.predict(X_test)

    # Get probability predictions if available
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = None

    # Calculate metrics
    metrics = {
        'Model': model_name,
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
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"    FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

    return metrics

def save_model_v2(model, model_name):
    """Save trained model to pickle file with v2 suffix."""
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    filename = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}_v2.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"  Saved model to {filename}")
    return filename

def print_comparison_table_v2(results):
    """Print a clean comparison table of all models."""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON (v2 - REALISTIC EVALUATION)")
    print("="*80)
    print("Note: Models evaluated on time-based test set (no data leakage)")
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
    print("SUMMARY STATISTICS (v2 - Realistic):")

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

def save_comparison_results_v2(results_df):
    """Save comparison results to CSV file with v2 suffix."""
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    filename = os.path.join(results_dir, 'model_comparison_v2.csv')
    results_df.to_csv(filename, index=False)

    print(f"\nComparison results saved to: {filename}")
    return filename

def compare_with_v1_results(results_df_v2):
    """Compare v2 results with previous (overfitted) v1 results."""
    print("\n" + "="*80)
    print("COMPARISON WITH PREVIOUS (OVERFITTED) RESULTS")
    print("="*80)

    try:
        # Load v1 results
        results_v1 = pd.read_csv('results/model_comparison.csv')

        print("\nV1 Results (with data leakage):")
        print(results_v1.to_string(index=False))

        print("\nV2 Results (no data leakage):")
        print(results_df_v2.to_string(index=False))

        print("\n" + "-"*80)
        print("PERFORMANCE DROP ANALYSIS:")

        # Compare key metrics
        for model in results_df_v2['Model'].unique():
            if model in results_v1['Model'].values:
                v1_row = results_v1[results_v1['Model'] == model].iloc[0]
                v2_row = results_df_v2[results_df_v2['Model'] == model].iloc[0]

                print(f"\n{model}:")
                for metric in ['Accuracy', 'F1-Score', 'ROC-AUC']:
                    if metric in v1_row and metric in v2_row:
                        v1_val = v1_row[metric]
                        v2_val = v2_row[metric]
                        if not pd.isna(v1_val) and not pd.isna(v2_val):
                            drop = v1_val - v2_val
                            print(f"  {metric}: {v1_val:.4f} → {v2_val:.4f} (drop: {drop:.4f})")

        print("\nKey Insight: Performance drop is expected and indicates")
        print("that previous results were overoptimistic due to data leakage.")

    except Exception as e:
        print(f"Could not load v1 results for comparison: {e}")

def main():
    """Main training pipeline v2."""
    print("="*80)
    print("PPC AD CLICK FRAUD DETECTION - MODEL TRAINING v2")
    print("="*80)
    print("Training on properly preprocessed data with NO DATA LEAKAGE")
    print("="*80)

    # Step 1: Load v2 data
    X_train, X_test, y_train, y_test = load_v2_data()

    # Step 2: Train models
    models = {}
    results = []

    # Decision Tree
    dt_model = train_decision_tree(X_train, y_train)
    models['Decision Tree'] = dt_model
    dt_metrics = evaluate_model(dt_model, 'Decision Tree', X_test, y_test)
    if dt_metrics:
        results.append(dt_metrics)
        save_model_v2(dt_model, 'decision_tree')

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf_model
    rf_metrics = evaluate_model(rf_model, 'Random Forest', X_test, y_test)
    if rf_metrics:
        results.append(rf_metrics)
        save_model_v2(rf_model, 'random_forest')

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    models['XGBoost'] = xgb_model
    if xgb_model:
        xgb_metrics = evaluate_model(xgb_model, 'XGBoost', X_test, y_test)
        if xgb_metrics:
            results.append(xgb_metrics)
            save_model_v2(xgb_model, 'xgboost')
    else:
        print("Skipping XGBoost (not available)")

    # LightGBM
    lgb_model = train_lightgbm(X_train, y_train)
    models['LightGBM'] = lgb_model
    if lgb_model:
        lgb_metrics = evaluate_model(lgb_model, 'LightGBM', X_test, y_test)
        if lgb_metrics:
            results.append(lgb_metrics)
            save_model_v2(lgb_model, 'lightgbm')
    else:
        print("Skipping LightGBM (not available)")

    # Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    models['Logistic Regression'] = lr_model
    lr_metrics = evaluate_model(lr_model, 'Logistic Regression', X_test, y_test)
    if lr_metrics:
        results.append(lr_metrics)
        save_model_v2(lr_model, 'logistic_regression')

    # Step 3: Print comparison table
    if results:
        results_df = print_comparison_table_v2(results)

        # Step 4: Save comparison results
        save_comparison_results_v2(results_df)

        # Step 5: Compare with v1 results
        compare_with_v1_results(results_df)

        # Additional analysis
        print("\n" + "="*80)
        print("MODEL TRAINING COMPLETE (v2)")
        print("="*80)

        print(f"\nTrained {len(models)} models:")
        for name, model in models.items():
            if model is not None:
                print(f"  - {name}")

        print(f"\nModels saved to 'models/' directory with '_v2' suffix")
        print(f"Results saved to 'results/model_comparison_v2.csv'")

        # Recommendation based on F1-Score (balanced metric)
        if 'F1-Score' in results_df.columns:
            best_f1_idx = results_df['F1-Score'].idxmax()
            best_model = results_df.loc[best_f1_idx, 'Model']
            best_f1 = results_df.loc[best_f1_idx, 'F1-Score']
            print(f"\nRECOMMENDATION (v2): {best_model} has the highest F1-Score ({best_f1:.4f})")

            # Note about realistic performance
            print(f"\nIMPORTANT: These results are REALISTIC (no data leakage)")
            print("Performance is lower than v1 because:")
            print("1. Time-based split (future data for testing)")
            print("2. Feature engineering using training data only")
            print("3. No information from test set used in training")
    else:
        print("\nNo models were successfully trained and evaluated.")

    return models, results

if __name__ == "__main__":
    models, results = main()