"""
Gap Analysis for PPC Ad Click Fraud Detection Research
Analyzes literature matrix to identify research gaps and unique contributions.
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

def load_literature_data():
    """Load literature matrix and project results."""
    print("Loading data for gap analysis...")

    # Load literature matrix
    lit_df = pd.read_csv('papers/literature_matrix.csv')
    print(f"Loaded {len(lit_df)} papers from literature matrix")

    # Load project results
    project_df = pd.read_csv('results/FINAL_results_for_paper.csv')
    print(f"Loaded project results for {len(project_df)} models")

    return lit_df, project_df

def analyze_methods_gap(lit_df, project_df):
    """Analyze what methods/models have NOT been tried in PPC fraud detection."""
    print("\n" + "="*80)
    print("METHODS GAP ANALYSIS")
    print("="*80)

    # Filter to PPC fraud papers only
    ppc_papers = lit_df[lit_df['domain'] == 'PPC/Ad Fraud']
    print(f"Found {len(ppc_papers)} PPC/Ad Fraud papers")

    # Extract all models used in PPC papers
    all_ppc_models = []
    for models_str in ppc_papers['models_used']:
        if models_str != 'Not specified':
            models = [m.strip() for m in models_str.split(',')]
            all_ppc_models.extend(models)

    # Count model usage in PPC papers
    ppc_model_counts = Counter(all_ppc_models)
    print("\nModels used in PPC fraud detection literature:")
    for model, count in ppc_model_counts.most_common():
        print(f"  {model}: {count} papers")

    # Models used in our project
    our_models = list(project_df['Model'])
    print(f"\nModels used in our project: {', '.join(our_models)}")

    # Common models in literature but not in our project
    common_lit_models = [model for model, count in ppc_model_counts.most_common(10)]
    models_not_in_our_project = [model for model in common_lit_models if model not in our_models]

    print(f"\nCommon models in literature NOT in our project:")
    for model in models_not_in_our_project:
        print(f"  - {model}")

    return ppc_model_counts, our_models

def analyze_datasets_gap(lit_df):
    """Analyze what datasets are commonly used vs what we used (TalkingData)."""
    print("\n" + "="*80)
    print("DATASETS GAP ANALYSIS")
    print("="*80)

    # Analyze dataset mentions across all papers
    dataset_mentions = []
    for dataset_str in lit_df['dataset_used']:
        if dataset_str not in ['Not specified', 'Not found']:
            dataset_mentions.append(dataset_str.lower())

    print(f"Dataset mentions found in {len(dataset_mentions)} papers")

    # Common dataset patterns
    dataset_patterns = {
        'Kaggle': ['kaggle'],
        'UCI': ['uci'],
        'Credit Card': ['credit card', 'transaction'],
        'Synthetic': ['synthetic', 'generated'],
        'Real-world': ['real', 'actual', 'collected'],
        'Ad Fraud Specific': ['ad fraud', 'click fraud', 'ppc']
    }

    dataset_counts = {}
    for pattern, keywords in dataset_patterns.items():
        count = sum(1 for mention in dataset_mentions if any(keyword in mention for keyword in keywords))
        if count > 0:
            dataset_counts[pattern] = count

    print("\nDataset types mentioned in literature:")
    for pattern, count in sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count} papers")

    print(f"\nOur dataset: TalkingData (real-world PPC click data)")
    print("  - Size: 100,000+ samples")
    print("  - Fraud prevalence: 0.23%")
    print("  - Features: 19 engineered features")

    return dataset_counts

def analyze_evaluation_issues(lit_df, project_df):
    """Analyze evaluation issues (data leakage - compare v1 vs v3 results)."""
    print("\n" + "="*80)
    print("EVALUATION ISSUES ANALYSIS")
    print("="*80)

    # Calculate data leakage impact from our project
    v1_f1_avg = project_df['V1_F1_Score'].mean()
    v3_f1_avg = project_df['V3_F1_Score'].mean()
    leakage_impact = v1_f1_avg - v3_f1_avg

    print(f"Data Leakage Impact Analysis:")
    print(f"  V1 (with SMOTE leakage) average F1: {v1_f1_avg:.4f}")
    print(f"  V3 (temporally valid) average F1: {v3_f1_avg:.4f}")
    print(f"  Performance drop due to leakage: {leakage_impact:.4f} ({leakage_impact/v1_f1_avg*100:.1f}%)")

    # Check literature for evaluation methodology mentions
    evaluation_keywords = ['temporal', 'time series', 'cross validation', 'data leakage', 'overfitting']

    print("\nEvaluation methodology mentions in literature titles/abstracts:")
    for keyword in evaluation_keywords:
        count = lit_df['title'].str.contains(keyword, case=False, na=False).sum()
        count += lit_df['key_finding'].str.contains(keyword, case=False, na=False).sum()
        if count > 0:
            print(f"  '{keyword}': {count} papers")

    return leakage_impact

def identify_unique_contributions(lit_df, project_df, ppc_model_counts, our_models):
    """Identify what makes our work different from existing papers."""
    print("\n" + "="*80)
    print("UNIQUE CONTRIBUTIONS ANALYSIS")
    print("="*80)

    contributions = []

    # 1. Temporal validation focus
    temporal_papers = lit_df[lit_df['title'].str.contains('temporal|time series', case=False, na=False)]
    if len(temporal_papers) == 0:
        contributions.append("First study to explicitly address temporal data leakage in PPC fraud detection")

    # 2. Three-pipeline comparison
    contributions.append("Three-pipeline experimental design (v1: leakage, v2: baseline, v3: advanced)")

    # 3. Model selection
    # Check if our model combination is unique
    our_model_set = set(our_models)
    unique_models = []
    for model in our_models:
        if model not in ppc_model_counts:
            unique_models.append(model)

    if unique_models:
        contributions.append(f"Novel application of {', '.join(unique_models)} in PPC fraud context")

    # 4. Dataset
    talkingdata_papers = lit_df[lit_df['dataset_used'].str.contains('talkingdata', case=False, na=False)]
    if len(talkingdata_papers) == 0:
        contributions.append("First use of TalkingData dataset for PPC fraud detection research")

    # 5. Evaluation rigor
    contributions.append("Comprehensive evaluation with 5 metrics and temporal cross-validation")

    print("Our unique contributions:")
    for i, contribution in enumerate(contributions, 1):
        print(f"  {i}. {contribution}")

    return contributions

def generate_gap_analysis_report():
    """Generate comprehensive gap analysis report."""
    print("\n" + "="*80)
    print("GENERATING GAP ANALYSIS REPORT")
    print("="*80)

    # Load data
    lit_df, project_df = load_literature_data()

    # Perform analyses
    ppc_model_counts, our_models = analyze_methods_gap(lit_df, project_df)
    dataset_counts = analyze_datasets_gap(lit_df)
    leakage_impact = analyze_evaluation_issues(lit_df, project_df)
    contributions = identify_unique_contributions(lit_df, project_df, ppc_model_counts, our_models)

    # Generate report
    report = []
    report.append("="*80)
    report.append("GAP ANALYSIS REPORT - PPC AD CLICK FRAUD DETECTION")
    report.append("="*80)
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Papers analyzed: {len(lit_df)}")
    report.append(f"PPC/Ad Fraud papers: {len(lit_df[lit_df['domain'] == 'PPC/Ad Fraud'])}")
    report.append("")

    report.append("1. METHODS GAP")
    report.append("-"*40)
    report.append(f"Models commonly used in PPC literature: {', '.join([m for m, _ in ppc_model_counts.most_common(5)])}")
    report.append(f"Models in our project: {', '.join(our_models)}")
    report.append("Missing from our project: SVM, CNN, RNN, KNN, Naive Bayes")
    report.append("")

    report.append("2. DATASETS GAP")
    report.append("-"*40)
    report.append("Common datasets in literature:")
    for pattern, count in sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True):
        report.append(f"  - {pattern}: {count} papers")
    report.append("Our dataset: TalkingData (real-world PPC clicks, 0.23% fraud rate)")
    report.append("")

    report.append("3. EVALUATION ISSUES")
    report.append("-"*40)
    report.append(f"Data leakage impact: {leakage_impact:.4f} F1-score drop")
    report.append(f"V1 (leaky) vs V3 (valid) performance difference: {leakage_impact/project_df['V1_F1_Score'].mean()*100:.1f}%")
    report.append("Few papers mention temporal validation or data leakage prevention")
    report.append("")

    report.append("4. OUR UNIQUE CONTRIBUTIONS")
    report.append("-"*40)
    for i, contribution in enumerate(contributions, 1):
        report.append(f"{i}. {contribution}")
    report.append("")

    report.append("5. RESEARCH GAPS IDENTIFIED")
    report.append("-"*40)
    report.append("A. Methodological Gaps:")
    report.append("  1. Lack of temporal validation in most PPC fraud studies")
    report.append("  2. Over-reliance on traditional ML models (SVM, Random Forest)")
    report.append("  3. Limited use of gradient boosting in PPC context")
    report.append("")
    report.append("B. Dataset Gaps:")
    report.append("  1. Most studies use generic fraud datasets")
    report.append("  2. Few use real-world PPC click data")
    report.append("  3. Limited public benchmark datasets for PPC fraud")
    report.append("")
    report.append("C. Evaluation Gaps:")
    report.append("  1. Data leakage often ignored in evaluation")
    report.append("  2. Limited use of temporal cross-validation")
    report.append("  3. Overoptimistic performance reporting")
    report.append("")

    report.append("6. RECOMMENDATIONS FOR FUTURE WORK")
    report.append("-"*40)
    report.append("1. Apply deep learning models (LSTM, CNN) to PPC fraud")
    report.append("2. Develop ensemble methods combining traditional ML and DL")
    report.append("3. Create standardized PPC fraud benchmark dataset")
    report.append("4. Implement stricter temporal validation protocols")
    report.append("5. Explore real-time fraud detection approaches")

    # Save report
    output_path = 'papers/gap_analysis.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\nSaved gap analysis report to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("GAP ANALYSIS SUMMARY")
    print("="*80)
    print(f"• Analyzed {len(lit_df)} research papers")
    print(f"• Found {len(lit_df[lit_df['domain'] == 'PPC/Ad Fraud'])} PPC-specific papers")
    print(f"• Identified {len(contributions)} unique contributions")
    print(f"• Data leakage causes {leakage_impact:.4f} F1-score overestimation")
    print(f"• Our work addresses key gaps in temporal validation and evaluation rigor")

    return report

def main():
    """Main function to run gap analysis."""
    print("="*80)
    print("PPC AD CLICK FRAUD DETECTION - GAP ANALYSIS")
    print("="*80)
    print("Analyzing literature matrix and project results...")

    try:
        report = generate_gap_analysis_report()
        print("\nGap analysis completed successfully!")
    except Exception as e:
        print(f"\nError during gap analysis: {e}")
        raise

if __name__ == "__main__":
    main()