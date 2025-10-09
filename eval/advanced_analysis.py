# advanced_analysis.py

import pandas as pd
from rouge_score import rouge_scorer
import textstat
from tqdm import tqdm
from pathlib import Path

# Loads summarisation results and evaluates using ANLP metrics
def run_advanced_analysis(input_filename='summarisation_results.csv'):
    print("Running advanced analysis")

    # Define project root relative to script location
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Build paths relative to project root
    filepath = PROJECT_ROOT / input_filename
    output_dir = PROJECT_ROOT / "outputs"
    output_path = output_dir / "advanced_evaluation_results.csv"

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        return

    # Initialise tools
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    evaluation_results = []

    # Filter out rows where translation failed
    df_successful = df[~df['generated_summary'].str.contains("Could not generate", na=False)].copy()

    print(f"Found {len(df_successful)} successful summaries to analyse")

    # Loop through each successful translation
    for index, row in tqdm(df_successful.iterrows(), total=len(df_successful), desc="Analysing Summaries"):
        original_text = str(row['original_text'])
        summary_text = str(row['generated_summary'])

        # Metric 1: ROUGE score for content overlap
        rouge_scores = scorer.score(original_text, summary_text)

        # Metric 2: Flesch-Kincaid readability scores for simplicity
        original_readability = textstat.flesch_kincaid_grade(original_text)
        summary_readability = textstat.flesch_kincaid_grade(summary_text)

        evaluation_results.append({
            'citation': row['citation'],
            'original_word_count': row['original_word_count'],
            'summary_word_count': row['summary_word_count'],
            'rougeL_precision': rouge_scores['rougeL'].precision,
            'rougeL_recall': rouge_scores['rougeL'].recall,
            'rougeL_fmeasure': rouge_scores['rougeL'].fmeasure,
            'original_readability_grade': original_readability,
            'summary_readability_grade': summary_readability,
            'readability_improvement': original_readability - summary_readability
        })

    # Save results to new file
    results_df = pd.DataFrame(evaluation_results)

    # Check output directory exists before saving, learned this the hard way
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")

    # Print overall average findings
    avg_readability_improvement = results_df['readability_improvement'].mean()
    avg_rougeL_fmeasure = results_df['rougeL_fmeasure'].mean()

    print(f"Average readability improvement: {avg_readability_improvement:.2f} grade levels")
    print(f"Average ROUGE-L F-measure: {avg_rougeL_fmeasure:.2%}")


if __name__ == "__main__":
    run_advanced_analysis()