FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY run_phase5.py prompts.py retrieval.py ./

# Model and vector DB are NOT baked into the image (15GB+).
# Mount them at runtime:
#   docker run -v $(pwd)/judaism-llm-qwen2.5-7b-merged:/app/model \
#              -v $(pwd)/chroma_db:/app/chroma_db -p 8000:8000 judaism-llm-rag
ENV MODEL_PATH=./model
ENV CHROMA_DIR=./chroma_db
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "run_phase5:app", "--host", "0.0.0.0", "--port", "8000"]
