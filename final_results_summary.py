"""
PPC Ad Click Fraud Detection - Final Results Summary
Creates clean comparison table for research paper.
"""

import pandas as pd
import numpy as np
import os

def load_and_clean_data():
    """Load v1 and v3 results and clean for final comparison."""
    print("Loading results files...")

    # Load v1 results (with SMOTE leakage)
    v1_df = pd.read_csv('results/model_comparison.csv')
    print(f"Loaded v1 results: {len(v1_df)} models")

    # Load v3 results (temporally valid)
    v3_df = pd.read_csv('results/model_comparison_v3.csv')
    print(f"Loaded v3 results: {len(v3_df)} models")

    # Standardize model names for merging
    v1_df['Model'] = v1_df['Model'].str.strip()
    v3_df['Model'] = v3_df['Model'].str.strip()

    # Ensure consistent model order
    model_order = ['Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM', 'Logistic Regression']

    # Create final comparison dataframe
    final_data = []

    for model in model_order:
        # Get v1 results
        v1_row = v1_df[v1_df['Model'] == model]
        v1_f1 = v1_row['F1-Score'].values[0] if not v1_row.empty else np.nan

        # Get v3 results
        v3_row = v3_df[v3_df['Model'] == model]

        if not v3_row.empty:
            v3_f1 = v3_row['F1-Score'].values[0]
            v3_precision = v3_row['Precision'].values[0]
            v3_recall = v3_row['Recall'].values[0]
            v3_roc_auc = v3_row['ROC-AUC'].values[0]
            v3_threshold = v3_row['Threshold'].values[0]
        else:
            v3_f1 = v3_precision = v3_recall = v3_roc_auc = v3_threshold = np.nan

        # Calculate improvement
        improvement = v3_f1 - v1_f1 if not np.isnan(v3_f1) and not np.isnan(v1_f1) else np.nan

        final_data.append({
            'Model': model,
            'V1_F1_Score': v1_f1,
            'V3_F1_Score': v3_f1,
            'V3_Precision': v3_precision,
            'V3_Recall': v3_recall,
            'V3_ROC_AUC': v3_roc_auc,
            'V3_Threshold': v3_threshold,
            'Improvement_V1_to_V3': improvement
        })

    final_df = pd.DataFrame(final_data)

    return final_df

def format_for_research_paper(final_df):
    """Format the dataframe for research paper presentation."""
    print("\n" + "="*80)
    print("FINAL RESULTS FOR RESEARCH PAPER")
    print("="*80)

    # Create formatted table
    print("\nTable: Model Performance Comparison (PPC Ad Click Fraud Detection)")
    print("-"*90)
    print(f"{'Model':<20} {'V1 F1-Score':>12} {'V3 F1-Score':>12} {'V3 Precision':>12} {'V3 Recall':>12} {'V3 ROC-AUC':>12}")
    print("-"*90)

    for _, row in final_df.iterrows():
        print(f"{row['Model']:<20} "
              f"{row['V1_F1_Score']:>12.4f} "
              f"{row['V3_F1_Score']:>12.4f} "
              f"{row['V3_Precision']:>12.4f} "
              f"{row['V3_Recall']:>12.4f} "
              f"{row['V3_ROC_AUC']:>12.4f}")

    print("-"*90)

    # Add summary statistics
    print("\nSummary Statistics:")
    print(f"Average V1 F1-Score: {final_df['V1_F1_Score'].mean():.4f}")
    print(f"Average V3 F1-Score: {final_df['V3_F1_Score'].mean():.4f}")
    print(f"Average Improvement: {final_df['Improvement_V1_to_V3'].mean():.4f}")

    # Best performing model in V3
    best_v3_idx = final_df['V3_F1_Score'].idxmax()
    best_model = final_df.loc[best_v3_idx, 'Model']
    best_f1 = final_df.loc[best_v3_idx, 'V3_F1_Score']

    print(f"\nBest Performing Model (V3): {best_model} (F1-Score: {best_f1:.4f})")

    return final_df

def save_final_results(final_df):
    """Save final results to CSV file."""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save to CSV
    output_path = 'results/FINAL_results_for_paper.csv'
    final_df.to_csv(output_path, index=False)

    print(f"\nSaved final results to: {output_path}")
    print(f"File contains {len(final_df)} models with {len(final_df.columns)} metrics")

    return output_path

def create_latex_table(final_df):
    """Generate LaTeX table for research paper."""
    print("\n" + "="*80)
    print("LaTeX TABLE FOR RESEARCH PAPER")
    print("="*80)

    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Model Performance Comparison for PPC Ad Click Fraud Detection}
\\label{tab:model-comparison}
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Model} & \\textbf{V1 F1-Score} & \\textbf{V3 F1-Score} & \\textbf{V3 Precision} & \\textbf{V3 Recall} & \\textbf{V3 ROC-AUC} \\\\
\\midrule
"""

    for _, row in final_df.iterrows():
        latex_table += f"{row['Model']} & {row['V1_F1_Score']:.4f} & {row['V3_F1_Score']:.4f} & {row['V3_Precision']:.4f} & {row['V3_Recall']:.4f} & {row['V3_ROC_AUC']:.4f} \\\\\n"

    latex_table += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item \\textbf{V1}: Results with SMOTE data leakage (overoptimistic)
\\item \\textbf{V3}: Temporally valid results with advanced techniques
\\item Best V3 performance in \\textbf{bold}
\\end{tablenotes}
\\end{table}"""

    print(latex_table)

    # Save LaTeX table to file
    latex_path = 'results/FINAL_results_latex_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)

    print(f"\nSaved LaTeX table to: {latex_path}")

    return latex_table

def main():
    """Main function to generate final results summary."""
    print("="*80)
    print("PPC AD CLICK FRAUD DETECTION - FINAL RESULTS SUMMARY")
    print("="*80)
    print("Creating clean comparison table for research paper")
    print("="*80)

    # Step 1: Load and clean data
    final_df = load_and_clean_data()

    # Step 2: Format for research paper
    formatted_df = format_for_research_paper(final_df)

    # Step 3: Save final results
    csv_path = save_final_results(formatted_df)

    # Step 4: Generate LaTeX table
    latex_table = create_latex_table(formatted_df)

    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("1. Loaded v1 (with SMOTE leakage) and v3 (temporally valid) results")
    print("2. Created clean comparison table with 5 models")
    print("3. Saved final results to: results/FINAL_results_for_paper.csv")
    print("4. Generated LaTeX table for research paper")
    print("5. Printed clean table ready to copy into research paper")

    # Key findings
    print("\nKey Findings:")
    print(f"- V1 results show overoptimistic performance due to data leakage")
    print(f"- V3 results represent realistic, temporally valid performance")
    print(f"- Best V3 model: {formatted_df.loc[formatted_df['V3_F1_Score'].idxmax(), 'Model']}")
    print(f"- Target F1-Score >= 0.5: {'ACHIEVED' if formatted_df['V3_F1_Score'].max() >= 0.5 else 'NOT ACHIEVED'}")

    if formatted_df['V3_F1_Score'].max() < 0.5:
        needed = 0.5 - formatted_df['V3_F1_Score'].max()
        print(f"  Needed improvement: +{needed:.4f}")

if __name__ == "__main__":
    main()