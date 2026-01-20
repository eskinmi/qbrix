from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from qbrixstore.config import PostgresSettings
from qbrixstore.postgres.models import Base


_engine = None
_session_factory = None


def init_db(
    settings: PostgresSettings | None = None,
) -> async_sessionmaker[AsyncSession]:
    global _engine, _session_factory

    if settings is None:
        settings = PostgresSettings()

    _engine = create_async_engine(
        settings.dsn,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,  # todo: check if these values are alright.
        max_overflow=10,
        pool_recycle=3600,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _session_factory


async def create_tables():
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _engine.begin() as conn:  # noqa
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _session_factory() as session:  # noqa
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
