from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.config.settings import settings
from app.database.models import Base

__all__ = ["engine", "AsyncSessionLocal", "Base", "get_db", "init_db"]

# Create async engine. settings.database_url is the canonical, sync-style URL
# (postgresql://...) used by Alembic; here it's rewritten to the asyncpg driver.
_async_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    _async_url,
    echo=settings.debug,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow
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
