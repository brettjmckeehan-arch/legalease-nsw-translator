# Use Python 3.12.11 slim image as base
FROM python:3.12.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download BART model and tokenizer to avoid runtime downloads
RUN python -c "from transformers import pipeline, BartTokenizer; \
    model_name = 'facebook/bart-large-cnn'; \
    print('Downloading BART model...'); \
    pipeline('summarization', model=model_name, device=-1); \
    BartTokenizer.from_pretrained(model_name); \
    print('BART model downloaded successfully')"

# Copy all application files
COPY . .

# Create necessary directories
RUN mkdir -p static src .streamlit

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Create entrypoint script
RUN printf '#!/bin/bash\n\
mkdir -p .streamlit\n\
cat > .streamlit/secrets.toml << EOF\n\
ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"\n\
OPENAI_API_KEY = "${OPENAI_API_KEY}"\n\
GOOGLE_API_KEY = "${GOOGLE_API_KEY}"\n\
EOF\n\
\n\
exec streamlit run streamlit_app.py \\\n\
    --server.port=${PORT:-8501} \\\n\
    --server.address=0.0.0.0 \\\n\
    --server.headless=true \\\n\
    --browser.gatherUsageStats=false\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Run the entrypoint script
CMD ["/app/entrypoint.sh"]