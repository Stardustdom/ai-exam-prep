from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.config.settings import settings
from app.database.models import Base

__all__ = ["engine", "AsyncSessionLocal", "Base", "get_db", "init_db"]


def _to_asyncpg(url: str) -> tuple:
    """settings.database_url is the canonical, sync-style URL (postgresql://...)
    used by Alembic (psycopg2) — rewritten here for the asyncpg driver. Managed
    Postgres providers (Neon included) put libpq-only query params like
    `sslmode`/`channel_binding` in their connection strings, meant for
    psycopg2/libpq clients; asyncpg's connect() raises TypeError on an
    unrecognized kwarg rather than ignoring it, so passed straight through
    they break every connection outright. Strip them and translate the SSL
    requirement into `connect_args` instead, which asyncpg does understand.
    A local dev URL has no sslmode param at all, so connect_args stays empty
    and behavior there is unchanged."""
    url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    sslmode = query.pop("sslmode", None)
    query.pop("channel_binding", None)
    clean_url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

    connect_args = {"ssl": "require"} if sslmode and sslmode != "disable" else {}
    return clean_url, connect_args


_async_url, _connect_args = _to_asyncpg(settings.database_url)

engine = create_async_engine(
    _async_url,
    echo=settings.debug,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    connect_args=_connect_args
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db() -> AsyncSession:
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Create tables that don't yet exist (dev convenience only — use Alembic migrations in production)"""
    from sqlalchemy import text
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
