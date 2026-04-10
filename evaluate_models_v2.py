"""
PPC Ad Click Fraud Detection - Model Evaluation v2
Comprehensive evaluation with honest performance analysis.
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

def load_v2_models_and_data():
    """Load v2 trained models and test data."""
    print("Loading v2 models and test data...")

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Load test data
    X_test = pd.read_csv('data/X_test_v2.csv')
    y_test = pd.read_csv('data/y_test_v2.csv')

    # Convert y_test to 1D array
    y_test = y_test.iloc[:, 0].values if y_test.shape[1] == 1 else y_test.values

    print(f"Test data: {X_test.shape[0]:,} samples, {X_test.shape[1]:,} features")
    print(f"Class distribution (REAL): {np.bincount(y_test)}")
    print(f"Fraud prevalence: {np.sum(y_test == 1) / len(y_test):.4f}")

    # Load all v2 models
    model_files = {
        'Decision Tree': 'models/decision_tree_v2.pkl',
        'Random Forest': 'models/random_forest_v2.pkl',
        'XGBoost': 'models/xgboost_v2.pkl',
        'LightGBM': 'models/lightgbm_v2.pkl',
        'Logistic Regression': 'models/logistic_regression_v2.pkl'
    }

    models = {}
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    models[name] = pickle.load(f)
                print(f"[✓] Loaded: {name}")
            except Exception as e:
                print(f"[X] Failed to load {name}: {e}")
        else:
            print(f"[X] File not found: {filepath}")

    return models, X_test, y_test

def generate_confusion_matrix_v2(model, model_name, X_test, y_test):
    """Generate and save confusion matrix for a model."""
    # Make predictions
    y_pred = model.predict(X_test)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate (0)', 'Fraud (1)'],
                yticklabels=['Legitimate (0)', 'Fraud (1)'])
    plt.title(f'Confusion Matrix - {model_name} (v2)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Save figure
    filename = f'results/confusion_matrix_{model_name.lower().replace(" ", "_")}_v2.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved confusion matrix: {filename}")
    return cm

def generate_roc_curve_v2(model, model_name, X_test, y_test):
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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name} (v2)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Save figure
    filename = f'results/roc_curve_{model_name.lower().replace(" ", "_")}_v2.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved ROC curve: {filename}")
    return fpr, tpr, roc_auc

def create_combined_roc_curves_v2(models, X_test, y_test):
    """Create one figure with all ROC curves overlaid."""
    print("\nCreating combined ROC curves visualization (v2)...")

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
    plt.title('ROC Curves Comparison - All Models (v2)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Save figure
    filename = 'results/roc_curves_comparison_v2.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved combined ROC curves: {filename}")

def create_metric_comparison_charts_v2(models, X_test, y_test):
    """Create bar charts comparing F1-Score and ROC-AUC across models."""
    print("\nCreating metric comparison charts (v2)...")

    # Calculate metrics for all models
    metrics_data = []

    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Get ROC-AUC
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = np.nan

        # Get confusion matrix for detailed metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Use fraud class (class 1) metrics for evaluation
        fraud_metrics = report.get('1', report.get('Fraud', report.get('1.0', {})))
        if not fraud_metrics:
            fraud_metrics = {'precision': 0, 'recall': 0, 'f1-score': 0}

        metrics_data.append({
            'Model': model_name,
            'Accuracy': report['accuracy'],
            'F1-Score': fraud_metrics.get('f1-score', 0),
            'ROC-AUC': roc_auc,
            'Precision': fraud_metrics.get('precision', 0),
            'Recall': fraud_metrics.get('recall', 0),
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Create F1-Score comparison chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_df['Model'], metrics_df['F1-Score'], color='skyblue')
    plt.title('F1-Score Comparison (Fraud Class) Across Models (v2)', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('F1-Score (Fraud Class)', fontsize=12)
    plt.ylim([0, 0.4])
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/f1_comparison_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved F1-Score comparison: results/f1_comparison_v2.png")

    # Create ROC-AUC comparison chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_df['Model'], metrics_df['ROC-AUC'], color='lightcoral')
    plt.title('ROC-AUC Comparison Across Models (v2)', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('ROC-AUC', fontsize=12)
    plt.ylim([0.8, 1.0])
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/roc_auc_comparison_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved ROC-AUC comparison: results/roc_auc_comparison_v2.png")

    return metrics_df

def generate_honest_evaluation_report(models, X_test, y_test, metrics_df):
    """Generate honest evaluation report comparing v1 and v2 results."""
    print("\nGenerating honest evaluation report...")

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("PPC AD CLICK FRAUD DETECTION - HONEST EVALUATION REPORT v2")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Test Set Size: {len(X_test):,} samples")
    report_lines.append(f"Fraud Prevalence: {np.sum(y_test == 1) / len(y_test):.4f}")
    report_lines.append(f"Number of Models Evaluated: {len(models)}")
    report_lines.append("")

    # Load v1 results for comparison
    try:
        v1_results = pd.read_csv('results/model_comparison.csv')
        v1_available = True
    except:
        v1_results = None
        v1_available = False
        report_lines.append("Note: Could not load v1 results for comparison")

    # Current (v2) performance
    report_lines.append("CURRENT PERFORMANCE (v2 - NO DATA LEAKAGE)")
    report_lines.append("-"*80)

    for model_name, model in models.items():
        report_lines.append(f"\n{model_name.upper()}")
        report_lines.append("-"*40)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'], output_dict=True)

        # Format the report for display
        report_lines.append(f"{'':20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
        report_lines.append("-"*60)

        for i, label in enumerate(['Legitimate', 'Fraud']):
            metrics = report[label]
            report_lines.append(f"{label:20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1-score']:>10.4f} {metrics['support']:>10.0f}")

        report_lines.append("")
        report_lines.append(f"{'accuracy':20} {'':10} {'':10} {report['accuracy']:>10.4f} {len(y_test):>10}")
        report_lines.append(f"{'macro avg':20} {report['macro avg']['precision']:>10.4f} {report['macro avg']['recall']:>10.4f} {report['macro avg']['f1-score']:>10.4f} {len(y_test):>10}")
        report_lines.append(f"{'weighted avg':20} {report['weighted avg']['precision']:>10.4f} {report['weighted avg']['recall']:>10.4f} {report['weighted avg']['f1-score']:>10.4f} {len(y_test):>10}")

        # Confusion matrix details
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        report_lines.append(f"Confusion Matrix Details:")
        report_lines.append(f"  True Negatives (TN): {tn:,}")
        report_lines.append(f"  False Positives (FP): {fp:,}")
        report_lines.append(f"  False Negatives (FN): {fn:,}")
        report_lines.append(f"  True Positives (TP): {tp:,}")

    # Comparison with v1 (if available)
    if v1_available:
        report_lines.append("\n" + "="*80)
        report_lines.append("COMPARISON WITH PREVIOUS RESULTS (v1 - WITH DATA LEAKAGE)")
        report_lines.append("="*80)

        report_lines.append("\nV1 Results (Overoptimistic due to data leakage):")
        for _, row in v1_results.iterrows():
            report_lines.append(f"\n{row['Model']}:")
            report_lines.append(f"  Accuracy:  {row['Accuracy']:.4f}")
            report_lines.append(f"  F1-Score:  {row['F1-Score']:.4f}")
            report_lines.append(f"  ROC-AUC:   {row['ROC-AUC']:.4f}")

        report_lines.append("\nV2 Results (Realistic - No data leakage):")
        for _, row in metrics_df.iterrows():
            report_lines.append(f"\n{row['Model']}:")
            report_lines.append(f"  Accuracy:  {row['Accuracy']:.4f}")
            report_lines.append(f"  F1-Score:  {row['F1-Score']:.4f}")
            report_lines.append(f"  ROC-AUC:   {row['ROC-AUC']:.4f}")

    # Performance drop analysis
    report_lines.append("\n" + "="*80)
    report_lines.append("PERFORMANCE DROP ANALYSIS")
    report_lines.append("="*80)

    report_lines.append("\nWhy performance dropped significantly:")
    report_lines.append("1. TIME-BASED SPLIT: Test set contains FUTURE data")
    report_lines.append("   - Models trained on past, tested on future")
    report_lines.append("   - Real-world scenario: predict future fraud")
    report_lines.append("")
    report_lines.append("2. NO DATA LEAKAGE in feature engineering:")
    report_lines.append("   - Frequency features calculated from TRAINING data only")
    report_lines.append("   - Test data features use training mappings")
    report_lines.append("   - Unseen values in test get frequency = 0")
    report_lines.append("")
    report_lines.append("3. REALISTIC CLASS IMBALANCE:")
    report_lines.append(f"   - Test set: {np.sum(y_test == 1):,} frauds out of {len(y_test):,} samples")
    report_lines.append(f"   - Fraud prevalence: {np.sum(y_test == 1) / len(y_test):.4f}")
    report_lines.append("   - SMOTE applied ONLY to training data")
    report_lines.append("")
    report_lines.append("4. CHALLENGE OF FUTURE PREDICTION:")
    report_lines.append("   - Fraud patterns may change over time")
    report_lines.append("   - New IPs, devices, apps appear in test set")
    report_lines.append("   - Models must generalize to unseen patterns")

    # Best model analysis
    report_lines.append("\n" + "="*80)
    report_lines.append("BEST MODEL ANALYSIS")
    report_lines.append("="*80)

    if not metrics_df.empty:
        # Find best models for each metric
        best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
        best_auc = metrics_df.loc[metrics_df['ROC-AUC'].idxmax()]
        best_precision = metrics_df.loc[metrics_df['Precision'].idxmax()]
        best_recall = metrics_df.loc[metrics_df['Recall'].idxmax()]

        report_lines.append(f"\nBest F1-Score: {best_f1['Model']} ({best_f1['F1-Score']:.4f})")
        report_lines.append(f"Best ROC-AUC: {best_auc['Model']} ({best_auc['ROC-AUC']:.4f})")
        report_lines.append(f"Best Precision: {best_precision['Model']} ({best_precision['Precision']:.4f})")
        report_lines.append(f"Best Recall: {best_recall['Model']} ({best_recall['Recall']:.4f})")

        # Trade-off analysis
        report_lines.append("\nTRADE-OFF ANALYSIS:")
        report_lines.append(f"1. {best_recall['Model']} has highest recall ({best_recall['Recall']:.4f})")
        report_lines.append("   - Catches most frauds but has many false positives")
        report_lines.append(f"2. {best_precision['Model']} has highest precision ({best_precision['Precision']:.4f})")
        report_lines.append("   - Few false alarms but misses many frauds")
        report_lines.append(f"3. {best_f1['Model']} has best balance (F1-Score: {best_f1['F1-Score']:.4f})")
        report_lines.append("   - Recommended for balanced approach")

        # Final recommendation
        report_lines.append("\n" + "="*80)
        report_lines.append("FINAL RECOMMENDATION")
        report_lines.append("="*80)

        report_lines.append(f"\nRecommended Model: {best_f1['Model']}")
        report_lines.append(f"  - F1-Score: {best_f1['F1-Score']:.4f}")
        report_lines.append(f"  - ROC-AUC: {best_auc['ROC-AUC']:.4f}")
        report_lines.append(f"  - Precision: {best_precision['Precision']:.4f}")
        report_lines.append(f"  - Recall: {best_recall['Recall']:.4f}")

        report_lines.append("\nDEPLOYMENT CONSIDERATIONS:")
        report_lines.append("1. Model performance is realistic but modest")
        report_lines.append("2. Consider ensemble methods to improve performance")
        report_lines.append("3. Regular retraining needed as fraud patterns evolve")
        report_lines.append("4. Threshold tuning may improve precision/recall trade-off")

    # Save report to file
    report_content = "\n".join(report_lines)
    report_filename = 'results/honest_evaluation_report_v2.txt'

    with open(report_filename, 'w') as f:
        f.write(report_content)

    print(f"\nSaved honest evaluation report: {report_filename}")

    # Also print to console
    print("\n" + "="*80)
    print("HONEST EVALUATION REPORT SUMMARY")
    print("="*80)
    print("\n".join(report_lines[:50]))  # Print first 50 lines
    print("\n[Full report saved to file]")

    return report_content

def main():
    """Main evaluation pipeline v2."""
    print("="*80)
    print("PPC AD CLICK FRAUD DETECTION - MODEL EVALUATION v2")
    print("="*80)
    print("Comprehensive evaluation with honest performance analysis")
    print("="*80)

    # Step 1: Load models and data
    models, X_test, y_test = load_v2_models_and_data()

    if not models:
        print("No models loaded. Exiting.")
        return

    # Step 2: Generate confusion matrices for all models
    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRICES")
    print("="*80)

    confusion_matrices = {}
    for model_name, model in models.items():
        print(f"\nProcessing {model_name}...")
        cm = generate_confusion_matrix_v2(model, model_name, X_test, y_test)
        confusion_matrices[model_name] = cm

    # Step 3: Generate ROC curves for all models
    print("\n" + "="*80)
    print("GENERATING ROC CURVES")
    print("="*80)

    roc_data = {}
    for model_name, model in models.items():
        print(f"\nProcessing {model_name}...")
        fpr, tpr, roc_auc = generate_roc_curve_v2(model, model_name, X_test, y_test)
        roc_data[model_name] = (fpr, tpr, roc_auc)

    # Step 4: Create combined ROC curves
    create_combined_roc_curves_v2(models, X_test, y_test)

    # Step 5: Create metric comparison charts
    metrics_df = create_metric_comparison_charts_v2(models, X_test, y_test)

    # Step 6: Generate honest evaluation report
    generate_honest_evaluation_report(models, X_test, y_test, metrics_df)

    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nGenerated visualizations in 'results/' directory:")
    print(f"  - Confusion matrices for {len(models)} models")
    print(f"  - Individual ROC curves for {len(models)} models")
    print(f"  - Combined ROC curves comparison")
    print(f"  - F1-Score and ROC-AUC comparison charts")
    print(f"  - Honest evaluation report (honest_evaluation_report_v2.txt)")

    print(f"\nModels evaluated: {list(models.keys())}")
    print(f"Test set size: {len(X_test):,} samples")
    print(f"Fraud prevalence: {np.sum(y_test == 1) / len(y_test):.4f}")

if __name__ == "__main__":
    main()