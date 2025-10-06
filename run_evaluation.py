# run_evaluation.py

import pandas as pd
import time
from tqdm import tqdm  # Progress bars to stop me going insane with anxiety
from src.summariser import initialise_summariser, summarise_text

# Translates corpus or sample, collects performance metrics and saves results
def run_mass_evaluation(sample_size=None):
    print("Starting test evaluation")
    
    # 1: Load model and tokeniser
    print("Step 1/5: Initialising translation model")
    summariser, tokeniser = initialise_summariser()
    if not summariser:
        print("Failed to initialise model. Aborting")
        return
    print("Model initialised")

    # 2: Load corpus
    print("Step 2/5: Loading nsw_corpus_final.parquet")
    try:
        df = pd.read_parquet('nsw_corpus_final.parquet')
        # Focus on legislation
        df_eval = df[df['source'] == 'nsw_legislation'].copy()
        print(f"Corpus loaded successfully. Found {len(df_eval)} legislation documents")
        
        df_eval = df_eval[df_eval['text'].str.split().str.len() < 8062] # Legislation in 75th percentile of word length
        print(f"Filtered to {len(df_eval)} documents under 8062 words")

    except FileNotFoundError:
        print("Error: nsw_corpus_final.parquet not found. Make sure it's in the same folder")
        return

    # 3: Sample for quicker test run
    if sample_size:
        print(f"Step 3/5: Using random sample of {sample_size} docs for this run")
        df_eval = df_eval.sample(n=sample_size, random_state=42) # Not making same reproducibility mistake as AT1
    else:
        print(f"Step 3/5: Using full dataset of {len(df_eval)} pieces of legislation.")


    # 4: Loop through legislation and generate translations
    print(f"Step 4/5: Processing {len(df_eval)} pieces of legislation... will take a while")
    results = []

    # Progress bar for loop
    for index, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Translating"):
        original_text = row['text']
        
        start_time = time.time()
        generated_summary = summarise_text(original_text, summariser, tokeniser)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        results.append({
            'citation': row.get('citation', 'N/A'),
            'original_word_count': len(original_text.split()),
            'summary_word_count': len(generated_summary.split()),
            'processing_time_secs': processing_time,
            'generated_summary': generated_summary,
            'original_text': original_text  # For easy review
        })

    # 5: Save results
    print("Step 5/5: Saving results to summarisation_results.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv('summarisation_results.csv', index=False, encoding='utf-8-sig')
    
    print("\nEvaluation complete")
    print(f"Results saved to summarisation_results.csv")
    print(f"\nAverage processing time: {results_df['processing_time_secs'].mean():.2f} seconds per document")

if __name__ == "__main__":
    # First test run on 25 docs, second on 10% sample of full NSW legislation corpus (2216 documents) (sample_size=222). Now all
    # Use run_mass_evaluation() to run on full dataset. To test, specify sample_size
    run_mass_evaluation()