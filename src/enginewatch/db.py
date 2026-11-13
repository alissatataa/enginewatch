"""SQLAlchemy models and engine factory."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Index, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Reading(Base):
    """One raw sensor snapshot for one engine at one operating cycle."""

    __tablename__ = "readings"
    __table_args__ = (UniqueConstraint("engine_id", "cycle", name="uq_reading_engine_cycle"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    engine_id: Mapped[int] = mapped_column(Integer, index=True)
    cycle: Mapped[int] = mapped_column(Integer)
    settings: Mapped[list] = mapped_column(JSON)
    sensors: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Prediction(Base):
    """RUL estimate produced for a reading. ``rul`` is NULL when no model is loaded."""

    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_engine_cycle", "engine_id", "cycle"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    engine_id: Mapped[int] = mapped_column(Integer)
    cycle: Mapped[int] = mapped_column(Integer)
    rul: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    engine_id: Mapped[int] = mapped_column(Integer, index=True)
    cycle: Mapped[int] = mapped_column(Integer)
    level: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def create_db_engine(url: str) -> Engine:
    kwargs: dict = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
    return create_engine(url, **kwargs)
