"""shared fixtures for integration tests."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from qbrixstore.postgres.models import Base, Tenant


@pytest.fixture(scope="session")
def event_loop_policy():
    """use default event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


@pytest_asyncio.fixture
async def db_engine():
    """create an in-memory sqlite database for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:  # noqa
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    """create a database session for testing."""
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def tenant_a(db_session: AsyncSession):
    """create tenant a for testing."""
    tenant = Tenant(id="tenant-a", name="Tenant A", slug="tenant-a")
    db_session.add(tenant)
    await db_session.flush()
    return tenant


@pytest_asyncio.fixture
async def tenant_b(db_session: AsyncSession):
    """create tenant b for testing."""
    tenant = Tenant(id="tenant-b", name="Tenant B", slug="tenant-b")
    db_session.add(tenant)
    await db_session.flush()
    return tenant
