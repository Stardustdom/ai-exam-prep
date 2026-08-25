# app/main.py
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.admin.main import app as admin_app
from app.bot.main import bot_app, setup_webhook, run_polling, shutdown_bot
from app.config.settings import settings
from app.database import init_db
from app.graph.checkpointer import init_checkpointer, close_checkpointer
import logging

logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="1.0.0")

# Mount admin app and bot webhook. Document processing runs in-process as a
# FastAPI BackgroundTask (see app.services.workers); the periodic quiz-expiry
# sweep that Celery beat used to drive is now the /internal/sweep-expired-quizzes
# route below, meant to be pinged on a schedule by an external cron service.
app.mount("/admin", admin_app)
app.mount("/webhook", bot_app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Exam Preparation Platform...")

    await init_db()
    logger.info("Database initialized")

    await init_checkpointer()
    logger.info("LangGraph checkpointer initialized")

    if settings.telegram_webhook_url:
        await setup_webhook()
    else:
        # No public URL configured (local dev) — fall back to polling.
        import asyncio
        asyncio.create_task(run_polling())
        logger.info("No TELEGRAM_WEBHOOK_URL set; started long-polling instead")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AI Exam Preparation Platform...")
    await shutdown_bot()
    await close_checkpointer()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.app_name, "environment": settings.app_env}


@app.post("/internal/sweep-expired-quizzes")
async def sweep_expired_quizzes_endpoint(x_sweep_secret: str = Header(default=None)):
    """Replaces Celery beat's 15s periodic sweep — meant to be called on a
    schedule by an external free cron pinger (e.g. cron-job.org) every few
    minutes, both to force-submit/evaluate any quiz past its deadline and
    (as a side effect, on a free-tier host that sleeps idle services) to
    keep this process from going to sleep between Telegram messages, as long
    as the ping interval stays under the host's idle-sleep threshold.
    Requires SWEEP_SECRET to be set — refuses to run an unauthenticated sweep
    rather than silently no-op, so a missing/misconfigured secret is loud."""
    if not settings.sweep_secret:
        raise HTTPException(status_code=503, detail="SWEEP_SECRET is not configured")
    if x_sweep_secret != settings.sweep_secret:
        raise HTTPException(status_code=403, detail="Invalid sweep secret")

    from app.workers.expiry import sweep_expired_quizzes
    count = await sweep_expired_quizzes()
    return {"status": "swept", "expired_quizzes_handled": count}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
