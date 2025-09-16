# src/summariser.py

import torch
from transformers import pipeline, BartTokenizer
import warnings
import math

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def initialise_summariser():
    """
    Initialises and returns the summarisation pipeline and its tokeniser, running on the CPU.
    """
    device = -1 # Force CPU
    print(f"Summariser will use device: CPU")
    
    model_name = "facebook/bart-large-cnn"
    try:
        summariser = pipeline("summarization", model=model_name, device=device)
        tokeniser = BartTokenizer.from_pretrained(model_name)
        print("Model and tokeniser initialised successfully.")
        return summariser, tokeniser
    except Exception as e:
        print(f"Error initialising model: {e}")
        return None, None

def summarise_text(text_to_summarise, summariser, tokeniser, max_len=150, min_len=40):
    """
    Summarises a given text, with robust error handling and a chunking mechanism
    inspired by the user's functional script.
    """
    if not text_to_summarise or not isinstance(text_to_summarise, str):
        return "Invalid input text provided."

    max_chunk_tokens = 1024 # BART's maximum input token limit
    
    try:
        tokens = tokeniser.encode(text_to_summarise)
        
        # --- PATH 1: For texts that do not require chunking ---
        if len(tokens) <= max_chunk_tokens:
            summary_list = summariser(
                text_to_summarise, 
                max_length=max_len, 
                min_length=min_len, 
                do_sample=False
            )
            # Robust check for an empty result from the model
            if summary_list and len(summary_list) > 0:
                return summary_list[0]['summary_text']
            else:
                return "Could not generate a summary for this document (model returned an empty result)."

        # --- PATH 2: For texts that require chunking ---
        summaries = []
        for i in range(0, len(tokens), max_chunk_tokens - 50): # 50 token overlap
            chunk_tokens = tokens[i:i + max_chunk_tokens]
            chunk_text = tokeniser.decode(chunk_tokens, skip_special_tokens=True)
            
            chunk_summary_list = summariser(
                chunk_text, 
                max_length=150,
                min_length=30, 
                do_sample=False
            )
            # Robust check for an empty result on EACH chunk
            if chunk_summary_list and len(chunk_summary_list) > 0:
                summaries.append(chunk_summary_list[0]['summary_text'])
        
        # Final check to ensure at least one chunk was successfully summarised
        if summaries:
            return " ".join(summaries)
        else:
            return "Could not generate a summary for this document (all text chunks failed to process)."

    except Exception as e:
        return f"An error occurred during summarisation: {e}"
