# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake tiktoken's BPE encoding into the image so chunking doesn't need
# outbound network access to OpenAI's CDN at runtime.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Copy application
COPY . .

# Create storage directory
RUN mkdir -p /app/storage

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Render (and most PaaS hosts) assign their own port at runtime via $PORT and
# route traffic to whatever the container actually listens on — a hardcoded
# --port 8000 would build fine but fail every health check once deployed
# there. Falls back to 8000 for docker-compose/local dev, where nothing sets
# $PORT. Shell form (not exec-form CMD ["..."]) is required for $PORT to
# actually expand — exec form passes it through literally, unexpanded.
EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}