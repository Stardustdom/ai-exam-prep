# app/services/workers.py
#
# Job execution: uploads enqueue a job and return immediately (spec section
# 5) rather than blocking an HTTP request on document parsing/embedding.
#
# Previously this ran on Celery + Redis, with a separate always-running
# worker process and beat scheduler (see git history / docker-compose.yml
# before this change). Replaced with FastAPI's BackgroundTasks — the job
# runs in this same process, right after the HTTP response is sent — for two
# reasons: (1) it drops Redis and two extra always-on containers entirely,
# which matters when the deployment target is a free-tier host that doesn't
# support background workers at all, not just "would rather not pay for
# one"; (2) it collapses "rebuild 3 images every code change" (app, worker,
# beat all shared this file) down to one. The tradeoff: no cross-process
# retry queue and no horizontal scaling — acceptable for this app's actual
# traffic (a single admin uploading resources, one Telegram bot instance).
# The periodic quiz-expiry sweep that Celery beat used to drive on a timer
# is now a plain HTTP endpoint (see app.main's /internal/sweep-expired-quizzes)
# that an external free cron pinger (e.g. cron-job.org) hits every few
# minutes — see README's Deployment section.
from typing import Awaitable, Callable
from fastapi import BackgroundTasks
import logging

logger = logging.getLogger(__name__)


async def _run_and_log(coro_func: Callable[..., Awaitable], *args, label: str) -> None:
    """BackgroundTasks swallows a task's return value and, depending on
    server config, can bury an exception in the general server log with no
    obvious link back to which job it was — label every run explicitly."""
    try:
        await coro_func(*args)
        logger.info(f"Background job completed: {label}")
    except Exception as exc:
        logger.error(f"Background job failed: {label}: {exc}")


class WorkerService:
    """Facade used by the admin API to enqueue jobs. FastAPI resolves the
    `background_tasks` constructor param automatically via Depends() the
    same way it would for a route function — no change needed at any call
    site that does `worker: WorkerService = Depends()`."""

    def __init__(self, background_tasks: BackgroundTasks):
        self.background_tasks = background_tasks

    async def queue_resource_processing(self, resource_id: str) -> None:
        from app.ingestion.resource_pipeline import process_resource
        self.background_tasks.add_task(
            _run_and_log, process_resource, resource_id, label=f"resource {resource_id}"
        )

    async def queue_blueprint_analysis(self, sample_paper_id: str) -> None:
        from app.ingestion.blueprint_pipeline import extract_sample_questions
        self.background_tasks.add_task(
            _run_and_log, extract_sample_questions, sample_paper_id, label=f"sample paper {sample_paper_id}"
        )

    async def queue_blueprint_generation(self, exam_id: str) -> None:
        from app.ingestion.blueprint_pipeline import generate_blueprint
        self.background_tasks.add_task(
            _run_and_log, generate_blueprint, exam_id, label=f"blueprint for exam {exam_id}"
        )
