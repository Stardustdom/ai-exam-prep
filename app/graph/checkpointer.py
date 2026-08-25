# app/graph/checkpointer.py
#
# The LangGraph checkpointer is what makes a Telegram conversation survive a
# server restart (spec section 16/23): every interrupt/resume step is
# persisted here, keyed by thread_id (== ChatSession.id), not held in
# process memory. Postgres in production; MemorySaver is a same-process
# fallback for local dev/tests where LANGGRAPH_CHECKPOINT_STORE=memory.
from contextlib import AsyncExitStack
from typing import Optional
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)

_checkpointer: Optional[BaseCheckpointSaver] = None
_exit_stack: Optional[AsyncExitStack] = None


async def init_checkpointer() -> BaseCheckpointSaver:
    """Call once at application startup."""
    global _checkpointer, _exit_stack
    if _checkpointer is not None:
        return _checkpointer

    if settings.langgraph_checkpoint_store == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        _exit_stack = AsyncExitStack()
        saver = await _exit_stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(settings.database_url)
        )
        await saver.setup()
        _checkpointer = saver
        logger.info("LangGraph checkpointer: Postgres")
    else:
        _checkpointer = MemorySaver()
        logger.warning("LangGraph checkpointer: in-memory (sessions will NOT survive a restart)")

    return _checkpointer


def get_checkpointer() -> BaseCheckpointSaver:
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized; call init_checkpointer() at application startup")
    return _checkpointer


async def close_checkpointer() -> None:
    global _checkpointer, _exit_stack
    if _exit_stack is not None:
        try:
            await _exit_stack.aclose()
        except Exception:
            pass  # the connection we're closing may itself already be dead
    _checkpointer = None
    _exit_stack = None


async def reinit_checkpointer() -> BaseCheckpointSaver:
    """Tear down and rebuild the checkpointer's connection pool. Unlike
    app.database's SQLAlchemy engine (pool_pre_ping=True heals this
    automatically), AsyncPostgresSaver opens one long-lived connection at
    startup with no equivalent self-healing — Neon (free tier: idle
    auto-suspend, and periodic admin-initiated connection termination even
    on an active project) killing that connection left every subsequent
    graph call failing until the whole process restarted. Called from
    app.bot.handlers on a detected connection error; safe to call
    concurrently from multiple in-flight requests since it only ever
    swaps the module-level reference, and get_checkpointer() picks up
    whatever's currently there."""
    await close_checkpointer()
    return await init_checkpointer()


_CONNECTION_ERROR_MARKERS = (
    "connection is closed", "connection was closed", "terminating connection",
    "connection reset", "connection refused", "ssl connection has been closed",
    "server closed the connection", "could not connect", "connection does not exist",
)


def is_connection_error(exc: BaseException) -> bool:
    """String-matched, not exception-type-matched: the underlying driver
    (asyncpg for app.database, psycopg for the checkpointer) each raise
    their own distinct exception classes for the same real-world failure
    (Neon killing a connection), and both wrap/re-raise through several
    layers (SQLAlchemy, psycopg_pool) — matching the message is far more
    reliable here than trying to enumerate every wrapper type."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _CONNECTION_ERROR_MARKERS)
