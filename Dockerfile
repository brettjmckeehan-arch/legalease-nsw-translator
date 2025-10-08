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
COPY requirements_prod.txt .

# Install Python dependencies - split into separate steps for better debugging
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU version with compatible versions
RUN pip install --no-cache-dir \
    torch==2.5.0 \
    torchvision==0.20.0 \
    torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install transformers and other key dependencies
RUN pip install --no-cache-dir transformers==4.56.1 tokenizers==0.22.0

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements_prod.txt

# Remove broken fitz package if it exists
RUN pip uninstall -y fitz || true

# Verify transformers is installed and pre-download BART model
RUN python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" && \
    python -c "from transformers import pipeline, BartTokenizer; \
    model_name = 'facebook/bart-large-cnn'; \
    print('Downloading BART model...'); \
    pipeline('summarization', model=model_name, device=-1); \
    BartTokenizer.from_pretrained(model_name); \
    print('BART model downloaded successfully')"

# Create entrypoint script for Streamlit with secrets management
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

# Copy application files LAST (changes frequently)
COPY . .

# Create necessary directories
RUN mkdir -p static src .streamlit

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the entrypoint script
CMD ["/app/entrypoint.sh"]