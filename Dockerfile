FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for faiss-cpu and psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for FAISS index persistence
RUN mkdir -p faiss_data

# Run the trading pipeline
CMD ["python", "-m", "pipeline.orchestrator"]
