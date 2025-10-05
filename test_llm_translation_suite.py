# test_llm_translation_suite.py

import json
import pandas as pd
import textstat
from tqdm import tqdm
import time
import evaluate
import os
from pathlib import Path # Import the Path object
from src.summariser import initialise_summariser, summarise_text
from src import llm_handler
from prompts import PROMPT_OPTIONS

# TEST CONFIG
PROMPTS_TO_TEST = list(PROMPT_OPTIONS.keys())

MODELS_TO_TEST = {
    "Anthropic": ["claude-3-opus-20240229", "claude-3-5-sonnet-20240620"],
    "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "Google": ["gemini-2.5-pro", "gemini-2.5-flash"]
}

# METRIC LOADING
bertscore = evaluate.load("bertscore")
rouge = evaluate.load("rouge")

# HELPER FUNCTIONS
def check_format(generated_text):
    if not isinstance(generated_text, str):
        return False
    required_headings = ["LEGISLATION:", "THE PROBLEM:", "HOW THIS AFFECTS YOU:", "YOUR OPTIONS:"]
    return all(heading in generated_text for heading in required_headings)

def run_generation_and_evaluation_suite(test_data_path="test_data.json"):
    # Load local model
    print("Initialising BART")
    summariser, tokeniser = initialise_summariser()
    if not summariser:
        print("Failed to initialise BART")
        return

    # Same output path as initial evals
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True) # Create folder if it doesn't exist
    output_csv = output_dir / "llm_generation_and_evaluation_results.csv"

    # Load test data
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    # Check for existing results to resume from
    completed_runs = set()
    if os.path.exists(output_csv):
        print(f"Found existing results file: '{output_csv}'. Resuming from last run.")
        existing_df = pd.read_csv(output_csv)
        for index, row in existing_df.iterrows():
            completed_runs.add((row['document_id'], row['prompt_name'], row['model_name']))
        print(f"Skipping {len(completed_runs)} completed runs.")
    else:
        print(f"No existing results file found. Starting a new run at '{output_csv}'.")

    total_runs = len(test_cases) * len(PROMPTS_TO_TEST) * sum(len(models) for models in MODELS_TO_TEST.values())
    
    print(f"\nStarting test suite: {len(test_cases)} documents, {len(PROMPTS_TO_TEST)} prompts, {sum(len(v) for v in MODELS_TO_TEST.values())} models. Total runs: {total_runs}")

    with tqdm(total=total_runs) as pbar:
        pbar.update(len(completed_runs))
        # 1. Loop through each raw document
        for case in test_cases:
            full_text = case['full_text']
            stage1_needed = any((case['id'], prompt_key, model_name) not in completed_runs 
                                for prompt_key in PROMPTS_TO_TEST 
                                for provider, models in MODELS_TO_TEST.items() 
                                for model_name in models)

            if not stage1_needed:
                continue

            # Stage 1: Run local BART summarisation once per document
            bart_summary = summarise_text(full_text, summariser, tokeniser)
            if not bart_summary or "error" in bart_summary.lower():
                pbar.update(len(PROMPTS_TO_TEST) * sum(len(models) for models in MODELS_TO_TEST.values()))
                continue

            stage2_input = f"ORIGINAL DOCUMENT:\n---\n{full_text}\n---\n\nSUMMARY OF DOCUMENT:\n---\n{bart_summary}\n---"

            for prompt_key in PROMPTS_TO_TEST:
                prompt_text = PROMPT_OPTIONS[prompt_key]
                for provider, models in MODELS_TO_TEST.items():
                    for model_name in models:
                        
                        if (case['id'], prompt_key, model_name) in completed_runs:
                            continue

                        pbar.set_description(f"Testing {case['id']} | {prompt_key} | {model_name}")
                        
                        start_time = time.time()
                        if provider == "Anthropic": generated_summary = llm_handler.call_anthropic(prompt_text, stage2_input, model_name)
                        elif provider == "OpenAI": generated_summary = llm_handler.call_openai(prompt_text, stage2_input, model_name)
                        else: generated_summary = llm_handler.call_google(prompt_text, stage2_input, model_name)
                        end_time = time.time()
                        latency = end_time - start_time
                        
                        if generated_summary:
                            bert_scores = bertscore.compute(predictions=[generated_summary], references=[bart_summary], lang="en")
                            rouge_scores = rouge.compute(predictions=[generated_summary], references=[bart_summary])
                            readability_score = textstat.flesch_kincaid_grade(generated_summary)
                            format_pass = check_format(generated_summary)
                        else:
                            bert_scores, rouge_scores = {}, {}
                            readability_score, format_pass = None, False
                        
                        result_data = {
                            "document_id": [case['id']], "prompt_name": [prompt_key], "api_provider": [provider],
                            "model_name": [model_name], "latency_seconds": [round(latency, 2)],
                            "bart_summary": [bart_summary], "generated_summary": [generated_summary if generated_summary else "API_CALL_FAILED"],
                            "bertscore_f1": [bert_scores.get('f1', [None])[0]], "rougeL": [rouge_scores.get('rougeL')],
                            "flesch_kincaid_grade": [readability_score], "format_pass": [format_pass]
                        }
                        result_df = pd.DataFrame(result_data)
                        result_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False, encoding='utf-8-sig')
                        
                        pbar.update(1)

    print("\nTEST SUITE COMPLETE")
    print(f"Results for all runs saved to '{output_csv}'.")

if __name__ == "__main__":
    run_generation_and_evaluation_suite()
    
    try:
        from llm_translation_suite_analysis import generate_visuals
        generate_visuals()
    except ImportError:
        print("\nCould not import 'llm_translation_suite_analysis'. Skipping visualisations")
    except Exception as e:
        print(f"\nAn error occurred during visualisation: {e}")