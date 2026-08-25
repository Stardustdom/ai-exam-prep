#!/bin/bash
# For a Unix-like shell (WSL/macOS/Linux) setting up a NATIVE (non-Docker) dev
# environment. `docker-compose up -d` is the recommended path (see README) —
# it needs none of this and includes Postgres+pgvector/Redis/worker/beat.
# This script assumes you already have a reachable Postgres with the
# pgvector extension and a Redis instance; it does not start either.

echo "========================================="
echo "AI Exam Preparation Platform Setup"
echo "========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
if [[ "$python_version" < "3.11" ]]; then
    echo "Error: Python 3.11 or higher is required"
    exit 1
fi

echo "Python version: $python_version ✓"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please edit .env file with your configuration"
fi

# Create storage directory
mkdir -p storage

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys and configuration"
echo "2. Start the application: uvicorn app.main:app --reload"
echo "3. Start Celery worker: celery -A app.services.workers.celery_app worker --loglevel=info"
echo "4. Start Celery beat (quiz-expiry sweep): celery -A app.services.workers.celery_app beat --loglevel=info"
echo "5. Admin API: http://localhost:8000/admin/api (login at POST /admin/api/auth/login)"
echo "6. Leave TELEGRAM_WEBHOOK_URL unset for local long-polling, or set it to your public HTTPS URL"
echo ""
echo "For Docker deployment: docker-compose up -d"