# AI Exam Preparation Platform

An AI-powered exam preparation system with an admin API for building exam
corpora and a Telegram bot for students. Curriculum extraction, exam
blueprint analysis, and question generation are grounded in admin-uploaded
study material via retrieval — the LLM is never asked to invent questions
from nothing. Orchestration for the multi-turn Telegram conversation uses
LangGraph with a persistent checkpointer, so an in-progress quiz survives a
server restart.

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy (async), PostgreSQL + pgvector
- **Orchestration**: LangGraph (interrupt/resume pattern, Postgres-backed checkpointer)
- **AI/ML**: OpenAI, Anthropic, or Gemini (provider-switchable — `LLM_PROVIDER` for generation,
  `EMBEDDING_PROVIDER` separately for embeddings)
- **Jobs**: FastAPI BackgroundTasks (document processing, blueprint generation) — in-process,
  no queue broker. The periodic quiz-expiry sweep is a plain HTTP endpoint
  (`/internal/sweep-expired-quizzes`) meant to be called on a schedule by an external cron
  service; see [Deployment](#deployment-free-tier-no-card-required) below.
- **Bot**: python-telegram-bot (long-polling for local dev, webhook for production)
- **Container**: Docker, Docker Compose (local dev)

## Prerequisites

- Docker Desktop (runs Postgres+pgvector, the API, and Adminer — see `docker-compose.yml`)
- A Gemini, OpenAI, or Anthropic API key, matching whichever `LLM_PROVIDER`/`EMBEDDING_PROVIDER`
  you choose (Gemini and OpenAI both offer embeddings; Anthropic doesn't, so it can only be
  `LLM_PROVIDER`, not `EMBEDDING_PROVIDER`)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)

Local Python is only needed if you want to run the app outside Docker; use
Python 3.12 (newer versions may lack prebuilt wheels for some dependencies).

## Setup

1. **Configure environment**

   ```bash
   cp .env.example .env
   ```

   Fill in `OPENAI_API_KEY` (and/or `ANTHROPIC_API_KEY`), `TELEGRAM_BOT_TOKEN`,
   `JWT_SECRET_KEY` and `SECRET_KEY` (any long random strings), and an admin
   password hash:

   ```bash
   python -c "from app.services.auth import hash_password; print(hash_password('choose-a-password'))"
   ```

   Paste the output into `ADMIN_PASSWORD_HASH` in `.env`.

2. **Start everything**

   ```bash
   docker-compose up -d
   ```

   This brings up Postgres (with the `pgvector` extension), the API (`app`),
   and Adminer (a DB browser at http://localhost:8080). Document processing
   and blueprint generation run in-process as FastAPI BackgroundTasks — no
   separate worker to start.

   The API creates its tables automatically on first boot. To use versioned
   Alembic migrations instead (recommended once you have real data to
   preserve across schema changes):

   ```bash
   docker-compose exec app alembic upgrade head
   ```

3. **Log in as admin**

   ```bash
   curl -X POST http://localhost:8000/admin/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "choose-a-password"}'
   ```

   Use the returned `access_token` as a Bearer token on every other
   `/admin/api/*` request.

4. **Create an exam and upload material**

   ```bash
   curl -X POST http://localhost:8000/admin/api/exams \
     -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
     -d '{"name": "Joint Entrance Examination", "short_name": "JEE"}'

   curl -X POST http://localhost:8000/admin/api/exams/<exam_id>/resources \
     -H "Authorization: Bearer <token>" -F "file=@physics_notes.pdf"

   curl -X POST http://localhost:8000/admin/api/exams/<exam_id>/sample-papers \
     -H "Authorization: Bearer <token>" -F "file=@jee_2023_paper.pdf"
   ```

   Uploads return immediately; processing (parsing, chunking, embedding,
   curriculum extraction / question extraction + blueprint generation) runs
   in the background. Poll `GET /admin/api/exams/<exam_id>/processing-status`,
   `GET /admin/api/exams/<exam_id>/curriculum`, and
   `GET /admin/api/exams/<exam_id>/blueprint` to watch it complete.

5. **Point Telegram at the bot**

   With `TELEGRAM_WEBHOOK_URL` unset, the app long-polls Telegram directly —
   nothing else to configure for local testing. For production, set
   `TELEGRAM_WEBHOOK_URL` to your public HTTPS base URL; the app registers
   the webhook (with a secret token Telegram must echo back) on startup.

   Message the bot; it walks through exam selection (button or free text,
   e.g. "jee" or "joint entrance examination"), question count, chapter/topic,
   and duration, then generates and runs the quiz entirely inside Telegram.

## Deployment (free tier, no card required)

`docker-compose.yml` is local-dev only. Production is meant to run on free-tier
hosts, which don't support a permanently-running worker process — that's the whole
reason background jobs moved from Celery+Redis to FastAPI BackgroundTasks (see
Technology Stack above), and why the quiz-expiry sweep is now a plain HTTP endpoint
instead of a scheduled beat task.

1. **Database — [Neon](https://neon.tech)**: free Postgres with the `pgvector`
   extension, no card required. Create a project, run `CREATE EXTENSION vector;`
   once connected, and use the connection string it gives you as `DATABASE_URL`
   (Neon's URLs need `sslmode=require`, which asyncpg/SQLAlchemy honor automatically
   from the URL).

2. **App — [Render](https://render.com)** free web service, no card required:
   - Connect the repo (or deploy from a Dockerfile — this project's `Dockerfile`
     works as-is)
   - Set env vars from `.env.example`: `DATABASE_URL` (from Neon), your LLM/embedding
     provider key(s), `TELEGRAM_BOT_TOKEN`, `ADMIN_PASSWORD_HASH`, `JWT_SECRET_KEY`,
     `SECRET_KEY`, and `SWEEP_SECRET` (generate with
     `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
   - Set **`TELEGRAM_WEBHOOK_URL`** to your Render service's public HTTPS URL
     (no trailing slash) — this switches the bot from long-polling to webhook mode,
     which is what the free tier's sleep-when-idle behavior needs

3. **Keep it warm + drive the quiz-expiry sweep — [cron-job.org](https://cron-job.org)**,
   free, no card: schedule a `POST` to
   `https://<your-render-url>/internal/sweep-expired-quizzes` with header
   `X-Sweep-Secret: <your SWEEP_SECRET>`, every 10 minutes. This both handles expired
   quizzes on schedule (Celery beat's old job) and, as a side effect, keeps Render's
   free service from fully spinning down between Telegram messages, since the ping
   interval stays under its idle-sleep threshold.

**What this gets you**: $0/month, no card, the bot works without your computer on.
**What it costs you**: if the cron pinger ever misses a beat and the service does go
to sleep, the *next* message has a ~30-60s cold-start delay before the bot replies —
after that it's responsive again. Everything else (quizzes, admin panel, blueprint
generation) works identically to local dev.

## Testing

```bash
pip install -r requirements.txt   # or: docker-compose exec app pip install -r requirements.txt
pytest
```

## Project Structure

```
app/
├── admin/        FastAPI admin API (exam/resource/sample-paper CRUD, blueprint & curriculum views)
├── agents/       The 9 LangGraph agents (session, exam/curriculum resolution, retrieval,
│                 question generation, quiz management, evaluation)
├── bot/          Telegram handlers + per-update dependency wiring
├── config/       Settings (env-driven)
├── database/     SQLAlchemy models, repositories, Alembic migrations
├── graph/        The LangGraph state machine (interrupt/resume) + checkpointer setup
├── ingestion/     Document parsing, chunking, curriculum extraction, sample-paper/blueprint pipeline
├── schemas/      Pydantic request/response models
├── services/     LLM/embedding provider abstraction, storage, vector search, semantic cache,
│                 Telegram client, auth, BackgroundTasks job wiring
└── workers/      The quiz-expiry sweep (app.workers.expiry) that runs independent of any user
                  message — triggered by /internal/sweep-expired-quizzes, see Deployment below
```

## Known limitations (honest status, not swept under the rug)

- **"Review Answers" / "View Explanations"** are wired up: each question already
  carries an LLM-generated `explanation` and `source_reference` from
  generation time (QuestionGeneratorAgent's own prompt asks for both,
  grounded in the retrieved chunks), and the results-screen buttons now
  display them — no extra LLM calls needed at review time.
- **A real admin dashboard exists** at `/admin/` (server-rendered, `app/admin/static/index.html`)
  — exam/resource/sample-paper management, curriculum browser, and blueprint view with a
  regenerate button that live-polls progress. Every capability is also reachable directly via
  `/admin/api/*` for scripting.
- **Resource versioning is structural but not yet enforced**: `Resource.version`
  exists and a quiz's `blueprint_version` is recorded, but nothing currently
  pins a quiz's retrieval to a specific resource version if the admin
  re-uploads a changed file with the same content.
- **Curriculum-to-chunk assignment** uses embedding similarity between each
  chunk and each chapter/topic name (fast, no extra LLM calls) rather than a
  dedicated classification pass — good enough to make retrieval filtering
  useful, but not as precise as a per-chunk LLM classification would be.
