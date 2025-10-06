import streamlit as st

# This will only work if you run it with streamlit
# streamlit run test_apis.py

st.write("Testing API Keys...")

if "ANTHROPIC_API_KEY" in st.secrets:
    st.success("✓ Anthropic API key found")
    st.write(f"Key starts with: {st.secrets['ANTHROPIC_API_KEY'][:10]}...")
else:
    st.error("✗ Anthropic API key missing")

if "OPENAI_API_KEY" in st.secrets:
    st.success("✓ OpenAI API key found")
    st.write(f"Key starts with: {st.secrets['OPENAI_API_KEY'][:10]}...")
else:
    st.error("✗ OpenAI API key missing")

if "GOOGLE_API_KEY" in st.secrets:
    st.success("✓ Google API key found")
    st.write(f"Key starts with: {st.secrets['GOOGLE_API_KEY'][:10]}...")
else:
    st.error("✗ Google API key missing")