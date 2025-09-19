# src/summariser.py

import torch
from transformers import pipeline, BartTokenizer
import warnings
import math

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Initialise and return summarisation pipeline and tokeniser on CPU
def initialise_summariser():
    device = -1 # Force CPU as couldn't get mine to work with GPU
    model_name = "facebook/bart-large-cnn"
    try:
        summariser = pipeline("summarization", model=model_name, device=device)
        tokeniser = BartTokenizer.from_pretrained(model_name)
        print("Model and tokeniser initialised successfully.")
        return summariser, tokeniser
    except Exception as e:
        print(f"Error initialising model: {e}")
        return None, None

# Summarises with long input chunking handling before second-level summary
def summarise_text(text_to_summarise, summariser, tokeniser):
    if not text_to_summarise or not isinstance(text_to_summarise, str):
        return "Invalid input text provided."

    max_chunk_length = 1024
    chunk_size = 800  
    chunk_overlap = 100

    try:
        tokens = tokeniser.encode(text_to_summarise)
        total_tokens = len(tokens)

        # If text <60 tokens, too short to summarise. Original text returned for transformation API step
        if total_tokens < 60:
            return text_to_summarise

        # PATH 1: For text long enough to summarise
        if total_tokens <= max_chunk_length:
            summary_result = summariser(text_to_summarise, max_length=150, min_length=40, do_sample=False)
            if summary_result and len(summary_result) > 0:
                 return summary_result[0]['summary_text']
            else:
                 return "Could not generate translation (model returned empty result)."

        # PATH 2: For long text
        summaries = []
        num_chunks = math.ceil((total_tokens - chunk_overlap) / (chunk_size - chunk_overlap))
        
        for i in range(num_chunks):
            start = i * (chunk_size - chunk_overlap)
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = tokeniser.decode(chunk_tokens, skip_special_tokens=True)
            
            chunk_summary_list = summariser(chunk_text, max_length=120, min_length=30, do_sample=False)
            if chunk_summary_list and len(chunk_summary_list) > 0:
                summaries.append(chunk_summary_list[0]['summary_text'])
        
        if not summaries:
            return "Could not generate translation from any text chunks"
            
        # Second-level summarisation to create coherent final summary
        combined_summary = " ".join(summaries)
        
        # If combined text short, do not re-summarise
        if len(tokeniser.encode(combined_summary)) <= 180:
             return combined_summary

        final_summary_list = summariser(combined_summary, max_length=180, min_length=60, do_sample=False)

        if final_summary_list and len(final_summary_list) > 0:
            return final_summary_list[0]['summary_text']
        else:
            # Fallback to joined summaries if final pass fails
            return combined_summary

    except Exception as e:
        return f"An unexpected error occurred during translation: {e}"