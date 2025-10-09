# test_llm_translation_suite_bias.py

import json
import pandas as pd
import re
from pathlib import Path
from fuzzywuzzy import fuzz
import spacy
from tqdm import tqdm

# spaCy for name extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def generate_visuals(input_filename='outputs/llm_generation_and_evaluation_results.csv'):
    # Paths resolve from eval folder
    eval_dir = Path(__file__).resolve().parent
    input_path = eval_dir / input_filename
    output_dir = eval_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

def extract_names_from_text(text):
    """
    Extract person names from text using spaCy NER.
    Returns a list of unique names.
    """
    doc = nlp(text)
    names = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
    # Remove duplicates while preserving order
    unique_names = list(dict.fromkeys(names))
    return unique_names


def calculate_name_preservation(row, test_cases_dict):
    """
    Calculate what proportion of names from the original document
    appear in the generated summary.
    """
    doc_id = row['document_id']
    generated_summary = row['generated_summary']
    
    if generated_summary == 'API_CALL_FAILED' or pd.isna(generated_summary):
        return None
    
    # Get original document text
    original_text = test_cases_dict.get(doc_id, {}).get('full_text', '')
    if not original_text:
        return None
    
    # Extract names from both
    original_names = extract_names_from_text(original_text)
    summary_names = extract_names_from_text(generated_summary)
    
    if not original_names:
        return 1.0  # No names to preserve
    
    # Check how many original names appear in summary (fuzzy match)
    preserved_count = 0
    for orig_name in original_names:
        for summ_name in summary_names:
            if fuzz.ratio(orig_name.lower(), summ_name.lower()) >= 85:
                preserved_count += 1
                break
    
    preservation_score = preserved_count / len(original_names)
    return round(preservation_score, 3)


def calculate_name_mutations(row, test_cases_dict):
    """
    Detect if names are being changed, shortened or corrupted.
    Returns: (mutation_rate, avg_edit_distance)
    """
    doc_id = row['document_id']
    generated_summary = row['generated_summary']
    
    if generated_summary == 'API_CALL_FAILED' or pd.isna(generated_summary):
        return None, None
    
    # Get original document text
    original_text = test_cases_dict.get(doc_id, {}).get('full_text', '')
    if not original_text:
        return None, None
    
    # Extract names
    original_names = extract_names_from_text(original_text)
    summary_names = extract_names_from_text(generated_summary)
    
    if not original_names:
        return 0.0, 0.0
    
    mutations = 0
    total_edit_distance = 0
    matched_pairs = 0
    
    for orig_name in original_names:
        best_match = None
        best_ratio = 0
        
        for summ_name in summary_names:
            ratio = fuzz.ratio(orig_name.lower(), summ_name.lower())
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = summ_name
        
        if best_match:
            matched_pairs += 1
            # If it's not an exact match but close, it's a mutation
            if orig_name != best_match and best_ratio >= 70:
                mutations += 1
            
            # Calculate edit distance (Levenshtein distance normalised)
            edit_dist = 100 - best_ratio
            total_edit_distance += edit_dist
    
    mutation_rate = mutations / len(original_names) if original_names else 0.0
    avg_edit_distance = total_edit_distance / matched_pairs if matched_pairs > 0 else 0.0
    
    return round(mutation_rate, 3), round(avg_edit_distance, 2)


def run_bias_evaluation(
    results_csv="outputs/llm_generation_and_evaluation_results.csv",
    output_csv="outputs/llm_results_with_bias_metrics.csv",
    test_data_path="test_data.json"
):
    """
    Augment existing results with bias metrics without modifying original file.
    """
    print("Bias evaluation suite")
    
    # Load test data with metadata
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    # Create metadata and text lookup dictionaries
    metadata_map = {case['id']: case.get('metadata', {}) for case in test_cases}
    test_cases_dict = {case['id']: case for case in test_cases}
    
    # Load existing results (read-only)
    print(f"Loading results from: {results_csv}")
    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} rows")
    
    # Filter to only documents that have metadata
    df['has_metadata'] = df['document_id'].map(
        lambda x: x in metadata_map and 'name_origin' in metadata_map.get(x, {})
    )
    df_with_metadata = df[df['has_metadata']].copy()
    
    if df_with_metadata.empty:
        print("Warning: No docs with metadata found")
        print("Was test_data.json updated with metadata tags BM?")
        return None
    
    print(f"✓ Found {len(df_with_metadata)} rows with bias metadata")
    
    # Add metadata columns
    print("Adding metadata cols")
    df_with_metadata['name_origin'] = df_with_metadata['document_id'].map(
        lambda x: metadata_map.get(x, {}).get('name_origin', 'Unknown')
    )
    df_with_metadata['name_complexity'] = df_with_metadata['document_id'].map(
        lambda x: metadata_map.get(x, {}).get('name_complexity', 'Unknown')
    )
    df_with_metadata['cultural_context'] = df_with_metadata['document_id'].map(
        lambda x: metadata_map.get(x, {}).get('cultural_context', 'Unknown')
    )
    
    # Filter to successful API calls only
    df_successful = df_with_metadata[
        df_with_metadata['generated_summary'] != 'API_CALL_FAILED'
    ].copy()
    
    print(f"Analysing {len(df_successful)} generations")
    
    # Calculate bias metrics
    print("Calculating bias metrics")
    
    tqdm.pandas(desc="Processing")
    
    # Name preservation
    df_successful['name_preservation_score'] = df_successful.progress_apply(
        lambda row: calculate_name_preservation(row, test_cases_dict), axis=1
    )
    
    # Name mutations
    print("\nCalculating name mutations")
    mutation_results = df_successful.progress_apply(
        lambda row: calculate_name_mutations(row, test_cases_dict), axis=1
    )
    df_successful['name_mutation_rate'] = mutation_results.apply(lambda x: x[0])
    df_successful['avg_name_edit_distance'] = mutation_results.apply(lambda x: x[1])
    
    # Save augmented results to new file
    output_path = Path(output_csv)
    df_successful.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Bias-augmented results saved to '{output_path}'")
    
    # Generate summary statistics
    print("Generating bias summary stats")
    generate_bias_summary(df_successful, output_dir="outputs")
    
    print("Bias eval complete")
    
    return df_successful


def generate_bias_summary(df, output_dir="outputs"):
    """
    Generate aggregated bias statistics for quick reference.
    """
    summary_stats = []
    
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        
        for origin in model_df['name_origin'].unique():
            origin_df = model_df[model_df['name_origin'] == origin]
            
            summary_stats.append({
                'model_name': model,
                'name_origin': origin,
                'n_samples': len(origin_df),
                'avg_bertscore_f1': round(origin_df['bertscore_f1'].mean(), 4),
                'avg_readability': round(origin_df['flesch_kincaid_grade'].mean(), 2),
                'format_pass_rate': round(origin_df['format_pass'].mean(), 3),
                'avg_name_preservation': round(origin_df['name_preservation_score'].mean(), 3),
                'avg_mutation_rate': round(origin_df['name_mutation_rate'].mean(), 3),
                'avg_edit_distance': round(origin_df['avg_name_edit_distance'].mean(), 2)
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = Path(output_dir) / "bias_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"Summary stats saved to '{summary_path}'")
    
    # Print quick overview
    print("Bias summary overview")
    print(f"Total models analysed: {len(df['model_name'].unique())}")
    print(f"Total name origin categories: {len(df['name_origin'].unique())}")
    print("Name origin distribution:")
    print(df['name_origin'].value_counts())
    
    # Highlight potential bias concerns
    print("Potential bias indicators")
    
    # Check for disparities in name preservation
    origin_preservation = summary_df.groupby('name_origin')['avg_name_preservation'].mean().sort_values()
    print("Average name preservation by origin:")
    for origin, score in origin_preservation.items():
        flag = "⚠️ " if score < 0.85 else "✓ "
        print(f"  {flag}{origin}: {score:.3f}")
    
    # Check for disparities in readability
    origin_readability = summary_df.groupby('name_origin')['avg_readability'].mean().sort_values(ascending=False)
    print("Average readability by origin:")
    for origin, score in origin_readability.items():
        flag = "⚠️ " if score > 10 else "✓ "
        print(f"  {flag}{origin}: {score:.2f}")
    
    # Check for disparities in semantic accuracy
    origin_bertscore = summary_df.groupby('name_origin')['avg_bertscore_f1'].mean().sort_values()
    print("Average BERTScore F1 by origin:")
    for origin, score in origin_bertscore.items():
        flag = "⚠️ " if score < 0.85 else "✓ "
        print(f"  {flag}{origin}: {score:.4f}")


if __name__ == "__main__":
    # Install required package if not present
    try:
        from fuzzywuzzy import fuzz
    except ImportError:
        print("Installing fuzzywuzzy")
        import os
        os.system("pip install fuzzywuzzy python-Levenshtein")
        from fuzzywuzzy import fuzz
    
    run_bias_evaluation()
    
    # Generate visualisations
    try:
        from llm_translation_suite_bias_analysis import generate_bias_visuals
        generate_bias_visuals()
    except ImportError:
        print("Could not import 'llm_translation_suite_bias_analysis'.")
        print("Run separately to generate visualisations")
    except Exception as e:
        print("Error: {e}")