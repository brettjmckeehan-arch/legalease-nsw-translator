# src/summariser.py

import torch
from transformers import pipeline, BartTokenizer
import warnings
import math

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def initialise_summariser():
    """
    Initialises the summarisation pipeline on the CPU to ensure compatibility.
    """
    # Force the model to use the CPU (-1). This is the critical line.
    device = -1
    
    model_name = "facebook/bart-large-cnn" 
    
    try:
        summariser = pipeline("summarization", model=model_name, device=device)
        tokeniser = BartTokenizer.from_pretrained(model_name)
        
        print("Summarisation model and tokeniser initialised successfully on CPU.")
        return summariser, tokeniser
        
    except Exception as e:
        print(f"Error initialising model: {e}")
        return None, None

def summarise_text(text_to_summarise, summariser, tokeniser):
    """Summarises text, handling long inputs by chunking."""
    if not text_to_summarise or not isinstance(text_to_summarise, str):
        return "Invalid input text provided."

    max_chunk_length = 1024
    chunk_size = 800  
    chunk_overlap = 100

    try:
        tokens = tokeniser.encode(text_to_summarise)
        total_tokens = len(tokens)

        if total_tokens < 60:
            return text_to_summarise

        if total_tokens <= max_chunk_length:
            summary_result = summariser(text_to_summarise, max_length=150, min_length=40, do_sample=False)
            if summary_result:
                return summary_result[0]['summary_text']
            else:
                return "Could not generate summary (model returned empty result)."

        summaries = []
        num_chunks = math.ceil((total_tokens - chunk_overlap) / (chunk_size - chunk_overlap))
        
        for i in range(num_chunks):
            start = i * (chunk_size - chunk_overlap)
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = tokeniser.decode(chunk_tokens, skip_special_tokens=True)
            
            chunk_summary_list = summariser(chunk_text, max_length=120, min_length=30, do_sample=False)
            if chunk_summary_list:
                summaries.append(chunk_summary_list[0]['summary_text'])
        
        if not summaries:
            return "Could not generate summary from any text chunks."
            
        combined_summary = " ".join(summaries)
        
        if len(tokeniser.encode(combined_summary)) <= 180:
            return combined_summary

        final_summary_list = summariser(combined_summary, max_length=180, min_length=60, do_sample=False)

        if final_summary_list:
            return final_summary_list[0]['summary_text']
        else:
            return combined_summary

    except Exception as e:
        return f"An unexpected error occurred during summarisation: {e}"