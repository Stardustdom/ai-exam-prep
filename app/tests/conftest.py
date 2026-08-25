import os
import sys
from pathlib import Path

# Dummy values so Settings() and any accidental service construction don't
# blow up on missing required config during test collection.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:test-not-real")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
