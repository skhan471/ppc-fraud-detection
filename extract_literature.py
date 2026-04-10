"""
Literature Review Extraction for PPC Ad Click Fraud Detection
Extracts metadata from research paper PDFs in the papers/ folder.
"""

import pdfplumber
import os
import re
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def extract_text_from_pdf(pdf_path, max_pages=3):
    """Extract text from first N pages of a PDF."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"  Error reading {pdf_path}: {e}")
        return ""

def extract_title(text):
    """Extract title from text (first long line)."""
    lines = text.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if len(line) > 20 and len(line) < 200:  # Reasonable title length
            # Skip common non-title lines
            if not line.lower().startswith(('abstract', 'introduction', 'keywords', '1.', 'i.', '©')):
                return line
    return "Not found"

def extract_year(text):
    """Extract publication year (4-digit number between 2015-2024)."""
    # Look for 4-digit years
    year_pattern = r'\b(20[0-2][0-9])\b'
    years = re.findall(year_pattern, text)

    # Filter to reasonable range
    valid_years = [int(y) for y in years if 2015 <= int(y) <= 2024]

    if valid_years:
        # Return the most common year or the first one
        return str(max(set(valid_years), key=valid_years.count))
    return "Not found"

def extract_authors(text, title):
    """Extract authors (text between title and abstract)."""
    lines = text.split('\n')
    title_line_idx = -1

    # Find title line
    for i, line in enumerate(lines):
        if title in line:
            title_line_idx = i
            break

    if title_line_idx == -1:
        return "Not found"

    # Collect lines after title until we hit abstract/introduction
    author_lines = []
    for i in range(title_line_idx + 1, min(title_line_idx + 10, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        # Stop when we hit section headers
        if line.lower().startswith(('abstract', 'introduction', 'keywords', '1.', 'i.')):
            break
        author_lines.append(line)

    return " ".join(author_lines) if author_lines else "Not found"

def extract_dataset(text):
    """Extract dataset information."""
    # Look for dataset mentions
    dataset_patterns = [
        r'dataset[:\s]+([^\.]+)',
        r'data[:\s]+([^\.]+)',
        r'collected[:\s]+([^\.]+)',
        r'used[:\s]+([^\.]+) data'
    ]

    for pattern in dataset_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return matches[0][:200]  # Limit length

    # Look for common dataset names
    common_datasets = ['kaggle', 'uci', 'ad fraud', 'click fraud', 'credit card', 'iot', 'network']
    for dataset in common_datasets:
        if dataset in text.lower():
            return f"Mentions {dataset}"

    return "Not specified"

def extract_models(text):
    """Extract machine learning models used."""
    model_keywords = [
        'Random Forest', 'SVM', 'XGBoost', 'Neural', 'LSTM',
        'Deep Learning', 'Decision Tree', 'Logistic', 'CNN', 'RNN', 'LightGBM',
        'Gradient Boosting', 'AdaBoost', 'KNN', 'Naive Bayes', 'Ensemble'
    ]

    found_models = []
    text_lower = text.lower()

    for model in model_keywords:
        if model.lower() in text_lower:
            found_models.append(model)

    return ", ".join(found_models) if found_models else "Not specified"

def extract_best_metric(text):
    """Extract best performance metric (accuracy/F1/AUC)."""
    # Look for percentage patterns near metric keywords
    patterns = [
        r'accuracy[^\n]*?(\d+\.?\d*%)',
        r'f1[^\n]*?(\d+\.?\d*%)',
        r'auc[^\n]*?(\d+\.?\d*%)',
        r'precision[^\n]*?(\d+\.?\d*%)',
        r'recall[^\n]*?(\d+\.?\d*%)'
    ]

    best_percentage = 0
    best_metric = "Not found"

    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            try:
                percentage = float(match.strip('%'))
                if percentage > best_percentage:
                    best_percentage = percentage
                    best_metric = f"{percentage:.1f}%"
            except:
                pass

    return best_metric

def extract_domain(text):
    """Extract domain/category of the paper."""
    domain_keywords = {
        'PPC/Ad Fraud': ['ppc', 'ad fraud', 'click fraud', 'pay-per-click', 'advertising'],
        'Credit Card Fraud': ['credit card', 'financial fraud', 'banking', 'transaction'],
        'IoT Security': ['iot', 'internet of things', 'sensor', 'device'],
        'Network Intrusion': ['network', 'intrusion', 'cyber', 'attack', 'anomaly'],
        'General ML': []  # Default category
    }

    text_lower = text.lower()
    for domain, keywords in domain_keywords.items():
        if domain == 'General ML':
            continue
        for keyword in keywords:
            if keyword in text_lower:
                return domain

    return 'General ML'

def extract_key_finding(text):
    """Extract key finding (first sentence after abstract or conclusion)."""
    # Find abstract or conclusion section
    sections = ['abstract', 'conclusion']

    for section in sections:
        section_idx = text.lower().find(section)
        if section_idx != -1:
            # Get text after section header
            after_section = text[section_idx + len(section):section_idx + 500]
            # Split into sentences
            sentences = re.split(r'[.!?]', after_section)
            if sentences:
                first_sentence = sentences[0].strip()
                if len(first_sentence) > 20:  # Reasonable sentence length
                    return first_sentence[:300]  # Limit length

    return "Not extracted"

def process_pdf_file(pdf_path):
    """Process a single PDF file and extract metadata."""
    filename = os.path.basename(pdf_path)
    print(f"Processing: {filename}")

    # Extract text from first 3 pages
    text = extract_text_from_pdf(pdf_path, max_pages=3)

    if not text:
        print(f"  Warning: No text extracted from {filename}")
        return None

    # Extract metadata
    title = extract_title(text)
    year = extract_year(text)
    authors = extract_authors(text, title)
    dataset = extract_dataset(text)
    models = extract_models(text)
    best_metric = extract_best_metric(text)
    domain = extract_domain(text)
    key_finding = extract_key_finding(text)

    return {
        'filename': filename,
        'title': title,
        'year': year,
        'authors': authors,
        'dataset_used': dataset,
        'models_used': models,
        'best_metric': best_metric,
        'domain': domain,
        'key_finding': key_finding
    }

def generate_summary(data):
    """Generate summary statistics."""
    print("\n" + "="*80)
    print("LITERATURE REVIEW SUMMARY")
    print("="*80)

    # Total papers
    total_papers = len(data)
    print(f"\nTotal papers processed: {total_papers}")

    # Papers by domain
    print("\nPapers by Domain Category:")
    print("-"*40)
    domain_counts = Counter([row['domain'] for row in data])
    for domain, count in domain_counts.most_common():
        percentage = (count / total_papers) * 100
        print(f"  {domain}: {count} papers ({percentage:.1f}%)")

    # Most common models
    print("\nMost Common Models Used Across All Papers:")
    print("-"*40)
    all_models = []
    for row in data:
        if row['models_used'] != 'Not specified':
            models = [m.strip() for m in row['models_used'].split(',')]
            all_models.extend(models)

    model_counts = Counter(all_models)
    for model, count in model_counts.most_common(10):
        percentage = (count / total_papers) * 100
        print(f"  {model}: {count} papers ({percentage:.1f}%)")

    # Year distribution
    print("\nPublication Year Distribution:")
    print("-"*40)
    year_counts = Counter([row['year'] for row in data if row['year'] != 'Not found'])
    for year, count in year_counts.most_common():
        print(f"  {year}: {count} papers")

def main():
    """Main function to process all PDFs and generate literature matrix."""
    print("="*80)
    print("LITERATURE REVIEW EXTRACTION FOR PPC AD CLICK FRAUD DETECTION")
    print("="*80)

    # Define paths
    papers_dir = 'papers'
    output_csv = 'papers/literature_matrix.csv'

    # Ensure output directory exists
    os.makedirs(papers_dir, exist_ok=True)

    # Get all PDF files
    pdf_files = []
    for file in os.listdir(papers_dir):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(papers_dir, file))

    print(f"Found {len(pdf_files)} PDF files in {papers_dir}")

    if not pdf_files:
        print("No PDF files found. Exiting.")
        return

    # Process each PDF
    all_data = []
    for pdf_file in pdf_files:
        metadata = process_pdf_file(pdf_file)
        if metadata:
            all_data.append(metadata)

    if not all_data:
        print("No data extracted from any PDF files.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSaved literature matrix to: {output_csv}")
    print(f"Extracted metadata for {len(df)} papers")

    # Generate summary
    generate_summary(all_data)

    # Print sample of extracted data
    print("\n" + "="*80)
    print("SAMPLE OF EXTRACTED DATA (first 3 papers):")
    print("="*80)
    for i, row in enumerate(df.head(3).to_dict('records')):
        print(f"\nPaper {i+1}: {row['filename']}")
        print(f"  Title: {row['title'][:80]}...")
        print(f"  Year: {row['year']}")
        print(f"  Domain: {row['domain']}")
        print(f"  Models: {row['models_used']}")
        print(f"  Best Metric: {row['best_metric']}")

if __name__ == "__main__":
    main()