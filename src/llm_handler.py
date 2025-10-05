# src/llm_handler.py

import streamlit as st
import anthropic
import openai
import google.generativeai as genai

def call_anthropic(prompt, summary_text, model):
    """Calls the Anthropic API to transform the summary."""
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("Anthropic API key is not set in your secrets.")
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=prompt,
            messages=[{"role": "user", "content": summary_text}]
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Anthropic API error: {e}")
        return None

def call_openai(prompt, summary_text, model):
    """Calls the OpenAI API to transform the summary."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key is not set in your secrets.")
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": summary_text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None

def call_google(prompt, summary_text, model):
    """Calls the Google Gemini API to transform the summary."""
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API key is not set in your secrets.")
        return None
    try:
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(
            model_name=model,
            system_instruction=prompt
        )
        response = model_instance.generate_content(summary_text)
        return response.text
    except Exception as e:
        st.error(f"Google Gemini API error: {e}")
        return None