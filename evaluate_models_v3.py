"""
PPC Ad Click Fraud Detection - Model Evaluation v3
Comprehensive evaluation with advanced analysis and three-way comparison.
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                           roc_curve, auc, precision_recall_curve, roc_auc_score)

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_v3_models_and_data():
    """Load v3 trained models and test data."""
    print("Loading v3 models and test data...")

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Load test data
    X_test = pd.read_csv('data/X_test_v3.csv')
    y_test = pd.read_csv('data/y_test_v3.csv')

    # Convert y_test to 1D array
    y_test = y_test.iloc[:, 0].values if y_test.shape[1] == 1 else y_test.values

    print(f"Test data: {X_test.shape[0]:,} samples, {X_test.shape[1]:,} features")
    print(f"Class distribution (REAL): {np.bincount(y_test)}")
    print(f"Fraud prevalence: {np.sum(y_test == 1) / len(y_test):.4f}")

    # Load all v3 models with metadata
    model_files = {
        'Decision Tree': 'models/decision_tree_v3.pkl',
        'Random Forest': 'models/random_forest_v3.pkl',
        'XGBoost': 'models/xgboost_v3.pkl',
        'LightGBM': 'models/lightgbm_v3.pkl',
        'Logistic Regression': 'models/logistic_regression_v3.pkl'
    }

    models = {}
    thresholds = {}
    cv_metrics = {}

    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)

                models[name] = model_data['model']
                thresholds[name] = model_data['threshold']
                cv_metrics[name] = {
                    'cv_mean_f1': model_data.get('cv_mean_f1', 0),
                    'cv_std_f1': model_data.get('cv_std_f1', 0)
                }

                print(f"[+] Loaded: {name} (threshold={thresholds[name]:.3f})")
            except Exception as e:
                print(f"[X] Failed to load {name}: {e}")
        else:
            print(f"[X] File not found: {filepath}")

    return models, thresholds, cv_metrics, X_test, y_test

def generate_confusion_matrix_v3(model, model_name, threshold, X_test, y_test):
    """Generate and save confusion matrix for a model with custom threshold."""
    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate (0)', 'Fraud (1)'],
                yticklabels=['Legitimate (0)', 'Fraud (1)'])
    plt.title(f'Confusion Matrix - {model_name} (v3, threshold={threshold:.3f})',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Save figure
    filename = f'results/confusion_matrix_{model_name.lower().replace(" ", "_")}_v3.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved confusion matrix: {filename}")
    return cm, y_pred

def generate_roc_curve_v3(model, model_name, threshold, X_test, y_test):
    """Generate ROC curve for a model."""
    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For models without predict_proba, use decision function
        y_pred_proba = model.decision_function(X_test)

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Create visualization
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    # Mark the optimal threshold point
    if hasattr(model, 'predict_proba'):
        # Find point on ROC curve closest to threshold
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
        # Find threshold index closest to our optimal threshold
        if len(thresholds_pr) > 0:
            idx = np.argmin(np.abs(thresholds_pr - threshold))
            if idx < len(fpr):
                plt.scatter(fpr[idx], tpr[idx], color='red', s=100,
                          label=f'Threshold={threshold:.3f}', zorder=5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name} (v3)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Save figure
    filename = f'results/roc_curve_{model_name.lower().replace(" ", "_")}_v3.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved ROC curve: {filename}")
    return fpr, tpr, roc_auc

def generate_precision_recall_curve_v3(model, model_name, threshold, X_test, y_test):
    """Generate Precision-Recall curve for a model."""
    if not hasattr(model, 'predict_proba'):
        return None, None, None

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate Precision-Recall curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    # Create visualization
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkgreen', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')

    # Mark the optimal threshold point
    if len(thresholds_pr) > 0:
        idx = np.argmin(np.abs(thresholds_pr - threshold))
        if idx < len(precision):
            plt.scatter(recall[idx], precision[idx], color='red', s=100,
                       label=f'Threshold={threshold:.3f}', zorder=5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name} (v3)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    # Save figure
    filename = f'results/pr_curve_{model_name.lower().replace(" ", "_")}_v3.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved Precision-Recall curve: {filename}")
    return precision, recall, pr_auc

def create_combined_visualizations_v3(models, thresholds, X_test, y_test):
    """Create combined ROC and PR curves for all models."""
    print("\nCreating combined visualizations (v3)...")

    # Combined ROC curves
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))

    for (model_name, model), color in zip(models.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, color=color,
                label=f'{model_name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - All Models (v3)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    filename = 'results/roc_curves_comparison_v3.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved combined ROC curves: {filename}")

    # Combined Precision-Recall curves (for models with predict_proba)
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(models)))

    for (model_name, model), color in zip(models.items(), colors):
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)

            plt.plot(recall, precision, lw=2, color=color,
                    label=f'{model_name} (AUC = {pr_auc:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison - All Models (v3)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)

    filename = 'results/pr_curves_comparison_v3.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved combined Precision-Recall curves: {filename}")

def create_metric_comparison_charts_v3(models, thresholds, cv_metrics, X_test, y_test):
    """Create comprehensive metric comparison charts."""
    print("\nCreating metric comparison charts (v3)...")

    # Calculate metrics for all models
    metrics_data = []

    for model_name, model in models.items():
        threshold = thresholds.get(model_name, 0.5)

        # Get predictions with threshold
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = None

        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Get ROC-AUC and PR-AUC
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
        else:
            roc_auc = np.nan
            pr_auc = np.nan

        # Get confusion matrix for detailed metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Use fraud class (class 1) metrics for evaluation
        fraud_metrics = report.get('1', report.get('Fraud', report.get('1.0', {})))
        if not fraud_metrics:
            fraud_metrics = {'precision': 0, 'recall': 0, 'f1-score': 0}

        # Get CV metrics
        cv_mean = cv_metrics.get(model_name, {}).get('cv_mean_f1', 0)
        cv_std = cv_metrics.get(model_name, {}).get('cv_std_f1', 0)

        metrics_data.append({
            'Model': model_name,
            'Threshold': threshold,
            'Accuracy': report['accuracy'],
            'F1-Score': fraud_metrics.get('f1-score', 0),
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc,
            'Precision': fraud_metrics.get('precision', 0),
            'Recall': fraud_metrics.get('recall', 0),
            'CV_F1_Mean': cv_mean,
            'CV_F1_Std': cv_std,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Create F1-Score comparison chart
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(metrics_df))
    bars = plt.bar(x_pos, metrics_df['F1-Score'], color='skyblue', alpha=0.7)

    # Add error bars for CV standard deviation
    plt.errorbar(x_pos, metrics_df['F1-Score'], yerr=metrics_df['CV_F1_Std'],
                fmt='none', ecolor='black', capsize=5, capthick=2)

    plt.title('F1-Score Comparison (Fraud Class) Across Models (v3)', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('F1-Score (Fraud Class)', fontsize=12)
    plt.xticks(x_pos, metrics_df['Model'], rotation=45, ha='right')
    plt.ylim([0, 0.6])

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        threshold = metrics_df.loc[i, 'Threshold']
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}\n(τ={threshold:.2f})', ha='center', va='bottom',
                fontsize=9)

    plt.tight_layout()
    plt.savefig('results/f1_comparison_v3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved F1-Score comparison: results/f1_comparison_v3.png")

    # Create Precision-Recall tradeoff chart
    plt.figure(figsize=(10, 8))
    for i, row in metrics_df.iterrows():
        plt.scatter(row['Recall'], row['Precision'], s=150, alpha=0.7,
                   label=f"{row['Model']} (F1={row['F1-Score']:.3f})")

        # Add model name near point
        plt.annotate(row['Model'], (row['Recall'], row['Precision']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Tradeoff Across Models (v3)', fontsize=16, fontweight='bold')
    plt.xlim([0, 1.05])
    plt.ylim([0, 0.5])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/pr_tradeoff_v3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved Precision-Recall tradeoff: results/pr_tradeoff_v3.png")

    return metrics_df

def generate_three_way_comparison_report(metrics_df_v3):
    """Generate comprehensive report comparing v1, v2, and v3 results."""
    print("\nGenerating three-way comparison report...")

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("PPC AD CLICK FRAUD DETECTION - THREE-WAY COMPARISON REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Test Set Size: {len(metrics_df_v3) if not metrics_df_v3.empty else 0} models")
    report_lines.append("")

    # Load v1 and v2 results for comparison
    try:
        results_v1 = pd.read_csv('results/model_comparison.csv')
        v1_available = True
    except:
        results_v1 = None
        v1_available = False
        report_lines.append("Note: Could not load v1 results for comparison")

    try:
        results_v2 = pd.read_csv('results/model_comparison_v2.csv')
        v2_available = True
    except:
        results_v2 = None
        v2_available = False
        report_lines.append("Note: Could not load v2 results for comparison")

    # Current (v3) performance
    report_lines.append("CURRENT PERFORMANCE (v3 - ADVANCED TRAINING)")
    report_lines.append("-"*80)

    if not metrics_df_v3.empty:
        for _, row in metrics_df_v3.iterrows():
            report_lines.append(f"\n{row['Model'].upper()}")
            report_lines.append("-"*40)
            report_lines.append(f"Threshold: {row['Threshold']:.3f}")
            report_lines.append(f"CV F1-Score: {row['CV_F1_Mean']:.4f} +/- {row['CV_F1_Std']:.4f}")
            report_lines.append("")
            report_lines.append(f"{'Metric':<15} {'Value':>10}")
            report_lines.append("-"*25)
            report_lines.append(f"{'Accuracy':<15} {row['Accuracy']:>10.4f}")
            report_lines.append(f"{'Precision':<15} {row['Precision']:>10.4f}")
            report_lines.append(f"{'Recall':<15} {row['Recall']:>10.4f}")
            report_lines.append(f"{'F1-Score':<15} {row['F1-Score']:>10.4f}")
            report_lines.append(f"{'ROC-AUC':<15} {row['ROC-AUC']:>10.4f}")
            report_lines.append(f"{'PR-AUC':<15} {row['PR-AUC']:>10.4f}")
            report_lines.append("")
            report_lines.append(f"Confusion Matrix Details:")
            report_lines.append(f"  True Positives (TP): {row['TP']:,}")
            report_lines.append(f"  False Positives (FP): {row['FP']:,}")
            report_lines.append(f"  True Negatives (TN): {row['TN']:,}")
            report_lines.append(f"  False Negatives (FN): {row['FN']:,}")

    # Three-way comparison table
    if v1_available and v2_available and not metrics_df_v3.empty:
        report_lines.append("\n" + "="*80)
        report_lines.append("THREE-WAY PERFORMANCE COMPARISON")
        report_lines.append("="*80)

        # Create comparison table for each model
        for model_name in metrics_df_v3['Model'].unique():
            report_lines.append(f"\n{model_name.upper()}")
            report_lines.append("-"*40)

            # Get results from each version
            v1_row = results_v1[results_v1['Model'] == model_name].iloc[0] if model_name in results_v1['Model'].values else None
            v2_row = results_v2[results_v2['Model'] == model_name].iloc[0] if model_name in results_v2['Model'].values else None
            v3_row = metrics_df_v3[metrics_df_v3['Model'] == model_name].iloc[0]

            report_lines.append(f"{'Version':<10} {'F1-Score':>10} {'Precision':>10} {'Recall':>10} {'ROC-AUC':>10}")
            report_lines.append("-"*50)

            if v1_row is not None:
                report_lines.append(f"{'v1':<10} {v1_row['F1-Score']:>10.4f} {v1_row.get('Precision', 'N/A'):>10} {v1_row.get('Recall', 'N/A'):>10} {v1_row['ROC-AUC']:>10.4f}")
            if v2_row is not None:
                report_lines.append(f"{'v2':<10} {v2_row['F1-Score']:>10.4f} {v2_row['Precision']:>10.4f} {v2_row['Recall']:>10.4f} {v2_row['ROC-AUC']:>10.4f}")
            report_lines.append(f"{'v3':<10} {v3_row['F1-Score']:>10.4f} {v3_row['Precision']:>10.4f} {v3_row['Recall']:>10.4f} {v3_row['ROC-AUC']:>10.4f}")

            # Calculate improvements
            if v2_row is not None:
                f1_improvement = v3_row['F1-Score'] - v2_row['F1-Score']
                precision_improvement = v3_row['Precision'] - v2_row['Precision']
                recall_improvement = v3_row['Recall'] - v2_row['Recall']

                report_lines.append(f"\nImprovement from v2 to v3:")
                report_lines.append(f"  F1-Score: {f1_improvement:+.4f}")
                report_lines.append(f"  Precision: {precision_improvement:+.4f}")
                report_lines.append(f"  Recall: {recall_improvement:+.4f}")

    # Performance evolution analysis
    report_lines.append("\n" + "="*80)
    report_lines.append("PERFORMANCE EVOLUTION ANALYSIS")
    report_lines.append("="*80)

    report_lines.append("\nV1 -> V2: Eliminating Data Leakage")
    report_lines.append("-"*40)
    report_lines.append("Key changes:")
    report_lines.append("1. Time-based split (80/20 chronological)")
    report_lines.append("2. Feature engineering using training data only")
    report_lines.append("3. No information from test set used in training")
    report_lines.append("4. SMOTE applied only to training data")
    report_lines.append("\nResult: Dramatic performance drop (expected)")
    report_lines.append("  - Shows previous results were overoptimistic")
    report_lines.append("  - Establishes realistic baseline")

    report_lines.append("\nV2 -> V3: Advanced Techniques for Improvement")
    report_lines.append("-"*40)
    report_lines.append("Key improvements:")
    report_lines.append("1. Threshold tuning for optimal precision-recall tradeoff")
    report_lines.append("2. Cost-sensitive learning with sophisticated class weights")
    report_lines.append("3. Temporal cross-validation for time-series stability")
    report_lines.append("4. Advanced feature engineering (rolling windows, entropy)")
    report_lines.append("5. Regularization and hyperparameter optimization")
    report_lines.append("\nGoal: Achieve F1-Score >= 0.5 while maintaining validity")

    # Target achievement analysis
    if not metrics_df_v3.empty:
        report_lines.append("\n" + "="*80)
        report_lines.append("TARGET ACHIEVEMENT ANALYSIS")
        report_lines.append("="*80)

        best_f1_idx = metrics_df_v3['F1-Score'].idxmax()
        best_model = metrics_df_v3.loc[best_f1_idx, 'Model']
        best_f1 = metrics_df_v3.loc[best_f1_idx, 'F1-Score']

        report_lines.append(f"\nBest Model: {best_model}")
        report_lines.append(f"Best F1-Score: {best_f1:.4f}")
        report_lines.append(f"Target: F1-Score >= 0.5")

        if best_f1 >= 0.5:
            report_lines.append("\n[+] TARGET ACHIEVED!")
            report_lines.append(f"  The v3 pipeline successfully achieved the target F1-Score.")
            report_lines.append(f"  Improvement over v2: {best_f1 - 0.3333:.4f}")
        else:
            report_lines.append(f"\n[-] TARGET NOT ACHIEVED")
            report_lines.append(f"  Current best F1-Score: {best_f1:.4f}")
            report_lines.append(f"  Needed improvement: {0.5 - best_f1:.4f}")
            report_lines.append(f"  Improvement over v2: {best_f1 - 0.3333:.4f}")

        # Recommendations for further improvement
        report_lines.append("\n" + "="*80)
        report_lines.append("RECOMMENDATIONS FOR FURTHER IMPROVEMENT")
        report_lines.append("="*80)

        report_lines.append("\nIf target not achieved, consider:")
        report_lines.append("1. Ensemble methods (stacking, voting, blending)")
        report_lines.append("2. More sophisticated feature engineering")
        report_lines.append("3. Deep learning approaches (LSTM for time series)")
        report_lines.append("4. Anomaly detection techniques")
        report_lines.append("5. External data sources (IP reputation, device fingerprinting)")
        report_lines.append("6. More aggressive threshold tuning")
        report_lines.append("7. Cost matrix optimization for business requirements")

    # Save report to file
    report_content = "\n".join(report_lines)
    report_filename = 'results/three_way_comparison_report_v3.txt'

    with open(report_filename, 'w') as f:
        f.write(report_content)

    print(f"\nSaved three-way comparison report: {report_filename}")

    # Also print summary to console
    print("\n" + "="*80)
    print("THREE-WAY COMPARISON REPORT SUMMARY")
    print("="*80)
    print("\n".join(report_lines[:50]))  # Print first 50 lines
    print("\n[Full report saved to file]")

    return report_content

def main():
    """Main evaluation pipeline v3."""
    print("="*80)
    print("PPC AD CLICK FRAUD DETECTION - MODEL EVALUATION v3")
    print("="*80)
    print("Comprehensive evaluation with advanced analysis")
    print("and three-way comparison (v1 vs v2 vs v3)")
    print("="*80)

    # Step 1: Load models and data
    models, thresholds, cv_metrics, X_test, y_test = load_v3_models_and_data()

    if not models:
        print("No models loaded. Exiting.")
        return

    # Step 2: Generate confusion matrices for all models
    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRICES")
    print("="*80)

    confusion_matrices = {}
    predictions = {}
    for model_name, model in models.items():
        print(f"\nProcessing {model_name}...")
        threshold = thresholds.get(model_name, 0.5)
        cm, y_pred = generate_confusion_matrix_v3(model, model_name, threshold, X_test, y_test)
        confusion_matrices[model_name] = cm
        predictions[model_name] = y_pred

    # Step 3: Generate ROC curves for all models
    print("\n" + "="*80)
    print("GENERATING ROC CURVES")
    print("="*80)

    roc_data = {}
    for model_name, model in models.items():
        print(f"\nProcessing {model_name}...")
        threshold = thresholds.get(model_name, 0.5)
        fpr, tpr, roc_auc = generate_roc_curve_v3(model, model_name, threshold, X_test, y_test)
        roc_data[model_name] = (fpr, tpr, roc_auc)

    # Step 4: Generate Precision-Recall curves for all models
    print("\n" + "="*80)
    print("GENERATING PRECISION-RECALL CURVES")
    print("="*80)

    pr_data = {}
    for model_name, model in models.items():
        print(f"\nProcessing {model_name}...")
        threshold = thresholds.get(model_name, 0.5)
        pr_result = generate_precision_recall_curve_v3(model, model_name, threshold, X_test, y_test)
        if pr_result:
            precision, recall, pr_auc = pr_result
            pr_data[model_name] = (precision, recall, pr_auc)

    # Step 5: Create combined visualizations
    create_combined_visualizations_v3(models, thresholds, X_test, y_test)

    # Step 6: Create metric comparison charts
    metrics_df = create_metric_comparison_charts_v3(models, thresholds, cv_metrics, X_test, y_test)

    # Step 7: Generate three-way comparison report
    generate_three_way_comparison_report(metrics_df)

    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE (v3)")
    print("="*80)
    print(f"\nGenerated visualizations in 'results/' directory:")
    print(f"  - Confusion matrices for {len(models)} models")
    print(f"  - Individual ROC curves for {len(models)} models")
    print(f"  - Individual Precision-Recall curves for {len(pr_data)} models")
    print(f"  - Combined ROC and PR curves comparisons")
    print(f"  - F1-Score comparison with CV error bars")
    print(f"  - Precision-Recall tradeoff chart")
    print(f"  - Three-way comparison report (three_way_comparison_report_v3.txt)")

    print(f"\nModels evaluated: {list(models.keys())}")
    print(f"Test set size: {len(X_test):,} samples")
    print(f"Fraud prevalence: {np.sum(y_test == 1) / len(y_test):.4f}")

    # Check target achievement
    if not metrics_df.empty:
        best_f1 = metrics_df['F1-Score'].max()
        best_model = metrics_df.loc[metrics_df['F1-Score'].idxmax(), 'Model']
        print(f"\nBest F1-Score: {best_f1:.4f} ({best_model})")
        print(f"Target: F1-Score >= 0.5")

        if best_f1 >= 0.5:
            print("[+] TARGET ACHIEVED!")
        else:
            print(f"[-] Target not achieved (needs +{0.5 - best_f1:.4f})")

if __name__ == "__main__":
    main()