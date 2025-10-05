# src/llm_handler.py

import streamlit as st
import anthropic
import openai
import google.generativeai as genai

def _handle_error(error_message):
    try:
        st.error(error_message)
    except Exception:
        print(error_message)

def call_anthropic(prompt, summary_text, model, stream=False):
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        _handle_error("Anthropic API key unavailable")
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        if stream:
            def stream_generator():
                with client.messages.stream(
                    model=model, max_tokens=2048, system=prompt, messages=[{"role": "user", "content": summary_text}]
                ) as stream:
                    for chunk in stream.text_stream:
                        yield chunk
            return stream_generator()
        else:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                system=prompt,
                messages=[{"role": "user", "content": summary_text}]
            )
            return response.content[0].text
    except Exception as e:
        _handle_error(f"Anthropic API error: {e}")
        return None

def call_openai(prompt, summary_text, model, stream=False):
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        _handle_error("OpenAI API key unavailable")
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        if stream:
            def stream_generator():
                stream_resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": summary_text}
                    ],
                    stream=True
                )
                for chunk in stream_resp:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            return stream_generator()
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": summary_text}
                ]
            )
            return response.choices[0].message.content
    except Exception as e:
        _handle_error(f"OpenAI API error: {e}")
        return None

def call_google(prompt, summary_text, model, stream=False):
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        _handle_error("Google API key unavailable")
        return None
    try:
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(
            model_name=model,
            system_instruction=prompt
        )
        if stream:
            def stream_generator():
                response = model_instance.generate_content(summary_text, stream=True)
                for chunk in response: # Missing text handler in stream chunks
                    if hasattr(chunk, 'text'):
                        yield chunk.text
            return stream_generator()
        else:
            response = model_instance.generate_content(summary_text)
            return response.text
    except Exception as e:
        _handle_error(f"Google Gemini API error: {e}")
        return None