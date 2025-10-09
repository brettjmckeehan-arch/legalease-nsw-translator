# llm_translation_suite_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_visuals(input_filename='outputs/llm_generation_and_evaluation_results_cleaned.csv'):
    
    # Paths
    project_root = Path(__file__).resolve().parent
    input_path = project_root / input_filename
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Results file not found at '{input_path}'.")
        print("Run 'test_llm_translationsuite.py' first to generate results")
        return

    # Filter out failed API calls for accurate metrics
    df_successful = df[df['generated_summary'] != 'API_CALL_FAILED'].copy()
    if df_successful.empty:
        print("No successful API calls found. Cannot plot")
        return

    # Plots
    sns.set_theme(style="whitegrid")
    
    # 1. Readability score by model
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df_successful,
        x='flesch_kincaid_grade',
        y='model_name',
        hue='model_name',  # I hate FutureWarnings
        legend=False,      # I hate FutureWarnings
        orient='h',
        errorbar=None
    )
    plt.title('Readability: Flesch-Kincaid grade by model', fontsize=16, weight='bold')
    plt.xlabel('Average grade level', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'readability_by_model.png', dpi=300)
    print(f"Saved: readability_by_model.png")

    # 2. Latency vs readability
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_successful,
        x='latency_seconds',
        y='flesch_kincaid_grade',
        hue='api_provider',
        style='api_provider',
        s=100,
        alpha=0.7
    )
    plt.title('Performance: Speed vs Simplicity', fontsize=16, weight='bold')
    plt.xlabel('Latency (seconds)', fontsize=12)
    plt.ylabel('Flesch-Kincaid grade level', fontsize=12)
    plt.legend(title='API provider')
    plt.tight_layout()
    plt.savefig(output_dir / 'latency_vs_readability.png', dpi=300) 
    print(f"Saved: latency_vs_readability.png")

    # 3. Factual accuracy (BERTScore) by model
    if 'bertscore_f1' in df_successful.columns:
        plt.figure(figsize=(12, 7))
        sns.barplot(
            data=df_successful,
            x='bertscore_f1',
            y='model_name',
            hue='model_name', 
            legend=False,  
            orient='h',
            errorbar=None
        )
        plt.title('Meaning preservation: BERTScore F1 by model', fontsize=16, weight='bold')
        plt.xlabel('Average BERTScore F1-measure', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(output_dir / 'bertscore_by_model.png', dpi=300)
        print(f"Saved: bertscore_by_model.png")
    else:
        print("\nSkipping BERTScore plot: 'bertscore_f1' col not found in CSV")

    # 4. Lexical similarity (ROUGE) by model
    if 'rougeL' in df_successful.columns:
        plt.figure(figsize=(12, 7))
        sns.barplot(
            data=df_successful,
            x='rougeL',
            y='model_name',
            hue='model_name',
            legend=False,
            orient='h',
            errorbar=None
        )
        plt.title('Lexical similarity: ROUGE-L score by model', fontsize=16, weight='bold')
        plt.xlabel('Average ROUGE-L F-measure', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(output_dir / 'rouge_by_model.png', dpi=300)
        print(f"Saved: rouge_by_model.png")
    else:
        print("\nSkipping ROUGE plot: 'rougeL' col not found in CSV")


    print("\nAnalysis complete. Visuals saved to 'outputs' folder")

if __name__ == "__main__":
    generate_visuals()