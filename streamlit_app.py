# streamlit_app.py

import streamlit as st
from src.summariser import initialise_summariser, summarise_text
import time
import fitz  # PyMuPDF
from pathlib import Path
import base64

# LOAD EXTERNAL CSS FILE
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")
load_css("static/style.css")

# HEADER

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image("static/logo.jpg", width=200)
with col2:
    st.markdown("<h1>LegalEase NSW</h1>", unsafe_allow_html=True)
    st.markdown("<h3>AI-Powered Legal Document and Legislation Summariser</h3>", unsafe_allow_html=True)
with col3:
    st.image("static/logo.jpg", width=200)

st.markdown("---")

# MODEL LOADING

@st.cache_resource
def load_model():
    """Loads the summarisation model and its tokeniser."""
    summariser, tokeniser = initialise_summariser()
    return summariser, tokeniser

summariser, tokeniser = load_model()

# MAIN APP BODY (TWO-COLUMN LAYOUT)

left_col, right_col = st.columns([1, 3])

# LEFT COLUMN: Information and disclaimer
with left_col:
    st.write(
        "Fed up with confusing legalese? Had enough of complicated legislation jargon? Does red tape make you see red? Add any document  and find out what they're really saying in plain English."
    )

    st.warning(
        "**Disclaimer:** This tool provides AI-generated summaries for informational purposes only and does not constitute legal advice. "
        "It is not a substitute for a qualified legal professional. Always consult a lawyer for advice."
    )

# RIGHT COLUMN: Input tabs
with right_col:
    tab1, tab2 = st.tabs(["Translate text", "Translate PDF document"])

    # Tab 1: Text input
    with tab1:
        user_text = st.text_area("Paste the legal document you want translated here:", height=250, label_visibility="collapsed")

        if st.button("Translate text"):
            # **FIX IS HERE**: Check if user_text exists AND has non-whitespace characters
            if user_text and user_text.strip():
                word_count = len(user_text.split())
                st.info(f"Processing {word_count} words of pasted text.")
                
                with st.spinner("LegalEase is translating your pasted text ... please wait."):
                    start_time = time.time()
                    generated_summary = summarise_text(user_text, summariser, tokeniser)
                    end_time = time.time()

                st.subheader("Your translation")
                st.success(f"Translation generated in {end_time - start_time:.2f} seconds.")
                st.write(generated_summary)
            else:
                st.warning("Please paste your text into the box to translate.")

    # Tab 2: PDF upload
    with tab2:
        uploaded_file = st.file_uploader("Upload a PDF file to translate.", type="pdf", label_visibility="collapsed")

        if uploaded_file is not None:
            if st.button("Translate PDF"):
                try:
                    pdf_bytes = uploaded_file.getvalue()
                    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                    
                    extracted_text = ""
                    for page in pdf_document:
                        extracted_text += page.get_text().replace('\n', ' ') + ' '

                    # **FIX IS HERE**: Also apply the robust check to the text extracted from the PDF
                    if extracted_text and extracted_text.strip():
                        st.info(f"Successfully extracted {len(extracted_text.split())} words from your PDF.")
                        
                        with st.spinner("LegalEase is translating your text ... please wait."):
                            start_time = time.time()
                            generated_summary = summarise_text(extracted_text, summariser, tokeniser)
                            end_time = time.time()

                        st.subheader("Your translation")
                        st.success(f"Translation generated in {end_time - start_time:.2f} seconds.")
                        st.write(generated_summary)

                        with st.expander("Click to view the full extracted text"):
                            st.text(extracted_text)
                    else:
                        st.warning("Could not find any text in the uploaded PDF.")

                except Exception as e:
                    st.error(f"An error occurred while processing your PDF: {e}")