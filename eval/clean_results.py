# clean_results.py

import pandas as pd
from pathlib import Path

def clean_csv_results(
    input_filename='outputs/llm_generation_and_evaluation_results.csv',
    output_filename='outputs/llm_generation_and_evaluation_results_cleaned.csv'
):
    """
    Loads the results CSV, removes entries for non-existent models,
    and saves a new, clean file.
    """
    print(f"Loading results from '{input_filename}'...")
    
    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"Error: Input file not found. Please ensure '{input_filename}' exists.")
        return

    original_rows = len(df)
    
    # Define the list of hypothetical models to remove
    models_to_remove = [
        "claude-4-opus-20250925",
        "claude-4-haiku-20250925"
    ]
    
    # Filter the DataFrame, keeping only the rows that are NOT in the removal list
    df_cleaned = df[~df['model_name'].isin(models_to_remove)]
    
    cleaned_rows = len(df_cleaned)
    rows_removed = original_rows - cleaned_rows
    
    # Save the new, cleaned DataFrame to a new file
    df_cleaned.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"Cleaning complete.")
    print(f"Removed {rows_removed} rows related to hypothetical models.")
    print(f"New file saved as '{output_filename}' with {cleaned_rows} rows.")

if __name__ == "__main__":
    clean_csv_results()