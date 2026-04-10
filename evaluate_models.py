"""
PPC Ad Click Fraud Detection - Model Evaluation and Visualization
Comprehensive evaluation of all trained models with visualizations.
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

def load_models_and_data():
    """Load all trained models and test data."""
    print("Loading models and test data...")

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Load test data
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')['is_attributed'].values

    print(f"Test data: {X_test.shape[0]:,} samples, {X_test.shape[1]:,} features")
    print(f"Class distribution: {np.bincount(y_test)}")

    # Load all models
    model_files = {
        'Decision Tree': 'models/decision_tree.pkl',
        'Random Forest': 'models/random_forest.pkl',
        'XGBoost': 'models/xgboost.pkl',
        'LightGBM': 'models/lightgbm.pkl',
        'Logistic Regression': 'models/logistic_regression.pkl'
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

def generate_confusion_matrix(model, model_name, X_test, y_test):
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
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Save figure
    filename = f'results/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved confusion matrix: {filename}")
    return cm

def generate_classification_report(model, model_name, X_test, y_test):
    """Generate classification report for a model."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'], output_dict=True)

    # Convert to DataFrame for easier handling
    report_df = pd.DataFrame(report).transpose()
    return report_df

def generate_roc_curve(model, model_name, X_test, y_test):
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
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Save figure
    filename = f'results/roc_curve_{model_name.lower().replace(" ", "_")}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved ROC curve: {filename}")
    return fpr, tpr, roc_auc

def create_combined_roc_curves(models, X_test, y_test):
    """Create one figure with all ROC curves overlaid."""
    print("\nCreating combined ROC curves visualization...")

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
    plt.title('ROC Curves Comparison - All Models', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Save figure
    filename = 'results/roc_curves_comparison.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved combined ROC curves: {filename}")

def create_combined_confusion_matrices(models, X_test, y_test):
    """Create one figure with all confusion matrices side by side."""
    print("\nCreating combined confusion matrices visualization...")

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))

    if n_models == 1:
        axes = [axes]

    for (model_name, model), ax in zip(models.items(), axes):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Legit', 'Fraud'],
                   yticklabels=['Legit', 'Fraud'])
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.02)

    # Save figure
    filename = 'results/confusion_matrices.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved combined confusion matrices: {filename}")

def create_metric_comparison_charts(models, X_test, y_test):
    """Create bar charts comparing F1-Score and ROC-AUC across models."""
    print("\nCreating metric comparison charts...")

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

        metrics_data.append({
            'Model': model_name,
            'F1-Score': report['weighted avg']['f1-score'],
            'ROC-AUC': roc_auc,
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall']
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Create F1-Score comparison chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_df['Model'], metrics_df['F1-Score'], color='skyblue')
    plt.title('F1-Score Comparison Across Models', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.ylim([0.8, 1.0])
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/f1_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved F1-Score comparison: results/f1_comparison.png")

    # Create ROC-AUC comparison chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_df['Model'], metrics_df['ROC-AUC'], color='lightcoral')
    plt.title('ROC-AUC Comparison Across Models', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('ROC-AUC', fontsize=12)
    plt.ylim([0.9, 1.01])
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/roc_auc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved ROC-AUC comparison: results/roc_auc_comparison.png")

    return metrics_df

def plot_feature_importance(models, X_test):
    """Plot feature importance from Random Forest model."""
    print("\nCreating feature importance visualization...")

    # Get Random Forest model
    rf_model = models.get('Random Forest')

    if rf_model is None or not hasattr(rf_model, 'feature_importances_'):
        print("  Random Forest model not available or doesn't have feature importances")
        return

    # Get feature importances
    importances = rf_model.feature_importances_
    feature_names = X_test.columns

    # Create DataFrame for sorting
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)

    # Take top 15 features
    top_features = feature_importance_df.tail(15)

    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_features['feature'], top_features['importance'], color='steelblue')
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 15 Feature Importances - Random Forest', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (importance, feature) in enumerate(zip(top_features['importance'], top_features['feature'])):
        plt.text(importance + 0.001, i, f'{importance:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved feature importance plot: results/feature_importance.png")

    # Also save feature importance data to CSV
    feature_importance_df.sort_values('importance', ascending=False).to_csv(
        'results/feature_importance.csv', index=False)
    print(f"  Saved feature importance data: results/feature_importance.csv")

    return feature_importance_df

def generate_evaluation_report(models, X_test, y_test, metrics_df, feature_importance_df):
    """Generate detailed evaluation report."""
    print("\nGenerating evaluation report...")

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("PPC AD CLICK FRAUD DETECTION - MODEL EVALUATION REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Test Set Size: {len(X_test):,} samples")
    report_lines.append(f"Number of Models Evaluated: {len(models)}")
    report_lines.append("")

    # Detailed classification reports for each model
    report_lines.append("DETAILED CLASSIFICATION REPORTS")
    report_lines.append("-"*80)

    for model_name, model in models.items():
        report_lines.append(f"\n{model_name.upper()}")
        report_lines.append("-"*40)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])
        report_lines.append(report)

    # Model comparison summary
    report_lines.append("\n" + "="*80)
    report_lines.append("MODEL COMPARISON SUMMARY")
    report_lines.append("="*80)

    # Find best models for each metric
    if not metrics_df.empty:
        best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
        best_auc = metrics_df.loc[metrics_df['ROC-AUC'].idxmax()]
        best_precision = metrics_df.loc[metrics_df['Precision'].idxmax()]
        best_recall = metrics_df.loc[metrics_df['Recall'].idxmax()]

        report_lines.append(f"\nBest F1-Score: {best_f1['Model']} ({best_f1['F1-Score']:.4f})")
        report_lines.append(f"Best ROC-AUC: {best_auc['Model']} ({best_auc['ROC-AUC']:.4f})")
        report_lines.append(f"Best Precision: {best_precision['Model']} ({best_precision['Precision']:.4f})")
        report_lines.append(f"Best Recall: {best_recall['Model']} ({best_recall['Recall']:.4f})")

    # Best model recommendation
    report_lines.append("\n" + "="*80)
    report_lines.append("RECOMMENDATION")
    report_lines.append("="*80)

    if not metrics_df.empty:
        # Recommend based on F1-Score (balanced metric)
        best_overall = metrics_df.loc[metrics_df['F1-Score'].idxmax()]

        report_lines.append(f"\nRecommended Model: {best_overall['Model']}")
        report_lines.append(f"Reason: Highest F1-Score ({best_overall['F1-Score']:.4f})")
        report_lines.append(f"Additional strengths:")
        report_lines.append(f"  - ROC-AUC: {best_overall['ROC-AUC']:.4f}")
        report_lines.append(f"  - Precision: {best_overall['Precision']:.4f}")
        report_lines.append(f"  - Recall: {best_overall['Recall']:.4f}")

        # Justification
        report_lines.append("\nJustification:")
        report_lines.append("1. F1-Score is the harmonic mean of precision and recall, making it")
        report_lines.append("   ideal for imbalanced classification problems like fraud detection.")
        report_lines.append("2. High ROC-AUC indicates excellent discrimination ability.")
        report_lines.append("3. Balanced performance across all metrics suggests robustness.")

    # Feature importance insights
    if feature_importance_df is not None:
        report_lines.append("\n" + "="*80)
        report_lines.append("FEATURE IMPORTANCE INSIGHTS (Random Forest)")
        report_lines.append("="*80)

        top_5 = feature_importance_df.sort_values('importance', ascending=False).head(5)
        report_lines.append("\nTop 5 Most Important Features:")
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            report_lines.append(f"  {i}. {row['feature']}: {row['importance']:.4f}")

        # Insights
        report_lines.append("\nKey Insights:")
        report_lines.append("1. Frequency-based features (ip_frequency, app_frequency) are highly important,")
        report_lines.append("   indicating that repeated clicks from same sources are strong fraud indicators.")
        report_lines.append("2. Combined features (ip_app_frequency, ip_device_frequency) capture")
        report_lines.append("   interaction patterns that are valuable for detection.")
        report_lines.append("3. Time-based features (click_hour, click_day) help identify temporal patterns")
        report_lines.append("   in fraudulent activity.")

    # Save report to file
    report_content = '\n'.join(report_lines)
    report_filename = 'results/evaluation_report.txt'
    with open(report_filename, 'w') as f:
        f.write(report_content)

    print(f"  Saved evaluation report: {report_filename}")
    return report_content

def main():
    """Main evaluation pipeline."""
    print("="*80)
    print("PPC AD CLICK FRAUD DETECTION - MODEL EVALUATION AND VISUALIZATION")
    print("="*80)

    # Step 1: Load models and test data
    models, X_test, y_test = load_models_and_data()

    if not models:
        print("No models loaded. Exiting.")
        return

    print(f"\nLoaded {len(models)} models for evaluation")

    # Step 2: Generate individual visualizations for each model
    print("\n" + "="*80)
    print("GENERATING INDIVIDUAL MODEL VISUALIZATIONS")
    print("="*80)

    all_reports = {}
    all_roc_data = {}

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")

        # Generate confusion matrix
        cm = generate_confusion_matrix(model, model_name, X_test, y_test)

        # Generate classification report
        report_df = generate_classification_report(model, model_name, X_test, y_test)
        all_reports[model_name] = report_df

        # Generate ROC curve
        fpr, tpr, roc_auc = generate_roc_curve(model, model_name, X_test, y_test)
        all_roc_data[model_name] = (fpr, tpr, roc_auc)

    # Step 3: Generate combined visualizations
    print("\n" + "="*80)
    print("GENERATING COMBINED VISUALIZATIONS")
    print("="*80)

    create_combined_roc_curves(models, X_test, y_test)
    create_combined_confusion_matrices(models, X_test, y_test)

    # Step 4: Generate metric comparison charts
    metrics_df = create_metric_comparison_charts(models, X_test, y_test)

    # Step 5: Plot feature importance
    feature_importance_df = plot_feature_importance(models, X_test)

    # Step 6: Generate comprehensive evaluation report
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE EVALUATION REPORT")
    print("="*80)

    report_content = generate_evaluation_report(models, X_test, y_test, metrics_df, feature_importance_df)

    # Step 7: Summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nGenerated visualizations saved to 'results/' directory:")
    print("  - Individual confusion matrices for each model")
    print("  - Individual ROC curves for each model")
    print("  - Combined ROC curves comparison")
    print("  - Combined confusion matrices")
    print("  - F1-Score comparison chart")
    print("  - ROC-AUC comparison chart")
    print("  - Feature importance plot (Random Forest)")
    print(f"  - Detailed evaluation report: results/evaluation_report.txt")

    return models, metrics_df, feature_importance_df

if __name__ == "__main__":
    main()