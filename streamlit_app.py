# streamlit_app.py

import streamlit as st
import time
from src.summariser import initialise_summariser, summarise_text
from src.pdf_handler import extract_text_from_pdf
from src import llm_handler
from prompts import PROMPT_OPTIONS
from pathlib import Path
import base64

# COOL TAB
st.set_page_config(
    page_title="LegalEase NSW",
    page_icon="⚖️",
    layout="wide"
)

# LOAD CSS FILE
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("static/style.css")


# HEADER
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image("static/logo.jpg", width=150)
with col2:
    st.markdown("<h1>LegalEase NSW</h1>", unsafe_allow_html=True)
    st.markdown("<h3>AI-Powered Legal Document and Legislation Summariser</h3>", unsafe_allow_html=True)
with col3:
    st.image("static/logo.jpg", width=150)

st.markdown("---")


# MODEL LOADING
@st.cache_resource
def load_model():
    """Loads the summarisation model and its tokeniser."""
    summariser, tokeniser = initialise_summariser()
    return summariser, tokeniser

summariser, tokeniser = load_model()


# LAYOUT DEFINITION
main_col, controls_col = st.columns([4, 1])


# MAIN APPLICATION AREA & CONTROLS

with main_col:
    # STATE 1: SHOW INPUTS
    if 'final_output' not in st.session_state or st.session_state.final_output is None:
        st.subheader("Enter your legal text")
        input_text = st.text_area("Paste the text from a legal document or legislation below:", height=300, label_visibility="collapsed")
        
        st.subheader("Or upload a PDF")
        uploaded_file = st.file_uploader("Upload a PDF document:", type=['pdf'], label_visibility="collapsed")

    # STATE 2: SHOW OUTPUTS
    else:
        st.subheader("Your translated summary")
        st.text_area("Final output", value=st.session_state.final_output, height=300, key="output_text", label_visibility="collapsed")
        
        if st.session_state.get('initial_summary'):
            with st.expander("Show initial summary from local BART model"):
                st.write(st.session_state.initial_summary)

with controls_col:
    st.subheader("Controls")
    prompt_key = st.selectbox(
        "Summary style",
        options=list(PROMPT_OPTIONS.keys()),
        key="prompt_key"
    )
    
    st.markdown("---")
    
    api_provider = st.selectbox("Choose AI provider", ("Anthropic", "OpenAI", "Google"), key="api_provider")

    if api_provider == "Anthropic":
        model_name = st.selectbox("Choose a model", ("claude-3-opus-20240229", "claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"), key="model_name")
    elif api_provider == "OpenAI":
        model_name = st.selectbox("Choose a model", ("gpt-5", "gpt-4o", "gpt-4-turbo"), key="model_name")
    else: # Google
        model_name = st.selectbox("Choose a model", ("gemini-2.5-pro-latest", "gemini-2.5-flash-latest", "gemini-1.5-pro-latest"), key="model_name")

    st.markdown("---")

    # Show correct button based on state
    if 'final_output' not in st.session_state or st.session_state.final_output is None:
        if st.button("Translate to plain English", type="primary", use_container_width=True):
            text_to_process = None
            if uploaded_file:
                text_to_process = extract_text_from_pdf(uploaded_file)
            elif input_text:
                text_to_process = input_text

            if isinstance(text_to_process, str) and text_to_process.strip():
                with st.spinner("Stage 1/2: Performing initial summary..."):
                    initial_summary = summarise_text(text_to_process, summariser, tokeniser)
                st.session_state.initial_summary = initial_summary

                if initial_summary and "error" not in initial_summary.lower():
                    with st.spinner(f"Stage 2/2: Rewriting with {api_provider}..."):
                        final_translation = llm_handler.call_anthropic(PROMPT_OPTIONS[prompt_key], initial_summary, model_name) if api_provider == "Anthropic" else \
                                           llm_handler.call_openai(PROMPT_OPTIONS[prompt_key], initial_summary, model_name) if api_provider == "OpenAI" else \
                                           llm_handler.call_google(PROMPT_OPTIONS[prompt_key], initial_summary, model_name)
                    st.session_state.final_output = final_translation
                    st.rerun()
                else:
                    st.error("The local summarisation failed.")
            else:
                st.warning("Please provide text or a PDF.")
    else:
        if st.button("⬅️ Translate another", use_container_width=True):
            st.session_state.final_output = None
            st.session_state.initial_summary = None
            st.rerun()


# DISCLAIMER
st.info("DISCLAIMER: THIS TOOL PROVIDES SUMMARIES FOR INFORMATIONAL PURPOSES ONLY AND DOES NOT CONSTITUTE LEGAL ADVICE. IT IS NOT A SUBSTITUTE FOR A QUALIFIED LEGAL PROFESSIONAL.")