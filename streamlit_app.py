# streamlit_app.py

import streamlit as st
from src.summariser import initialise_summariser, summarise_text
import time
import fitz  # PyMuPDF
from pathlib import Path

# --- App Title and Description ---
st.title("LegalEase: Law jargon translator")
st.write(
    "Fed up with confusing legalese? Upload your PDF document or paste text directly and find out what it means in simple English."
    "NOTE: This app is no substitute for legal advice."
)

# --- Caching the Model ---
@st.cache_resource
def load_model():
    """Loads the summarisation model and its tokeniser."""
    summariser, tokeniser = initialise_summariser()
    return summariser, tokeniser

summariser, tokeniser = load_model()

# --- Create Tabs for Different Input Methods ---
tab1, tab2 = st.tabs(["Summarise a PDF Document", "Summarise Pasted Text"])

# --- Tab 1: PDF Uploader ---
with tab1:
    st.header("Upload Your Document")
    uploaded_file = st.file_uploader("Choose a PDF file to summarise...", type="pdf")

    if uploaded_file is not None:
        try:
            pdf_bytes = uploaded_file.getvalue()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            extracted_text = ""
            for page in pdf_document:
                extracted_text += page.get_text()

            st.info(f"Successfully extracted {len(extracted_text.split())} words from the PDF.")
            
            with st.spinner("The AI is summarising your document... Please wait."):
                start_time = time.time()
                generated_summary = summarise_text(extracted_text, summariser, tokeniser)
                end_time = time.time()

            st.subheader("Generated Summary")
            st.success(f"Summary generated in {end_time - start_time:.2f} seconds.")
            st.write(generated_summary)

            with st.expander("Click to view the full extracted text"):
                st.text(extracted_text)

        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")

# --- Tab 2: Text Input ---
with tab2:
    st.header("Paste Your Text")
    user_text = st.text_area("Paste the legal text you want to summarise here (e.g., from a contract or letter):", height=250)

    if st.button("Summarise Pasted Text"):
        if user_text:
            word_count = len(user_text.split())
            st.info(f"Processing {word_count} words of pasted text.")
            
            with st.spinner("The AI is summarising your text... Please wait."):
                start_time = time.time()
                generated_summary = summarise_text(user_text, summariser, tokeniser)
                end_time = time.time()

            st.subheader("Generated Summary")
            st.success(f"Summary generated in {end_time - start_time:.2f} seconds.")
            st.write(generated_summary)
        else:
            st.warning("Please paste some text into the box to summarise.")