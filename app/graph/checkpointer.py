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
        await _exit_stack.aclose()
    _checkpointer = None
    _exit_stack = None
