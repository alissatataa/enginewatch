"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ReadingIn(BaseModel):
    engine_id: int = Field(ge=1)
    cycle: int = Field(ge=1)
    settings: list[float] = Field(min_length=3, max_length=3, description="3 operating settings")
    sensors: dict[str, float] = Field(min_length=1, description="e.g. {'s2': 642.1, ...}")


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    engine_id: int
    cycle: int
    level: str
    message: str
    created_at: datetime


class IngestOut(BaseModel):
    engine_id: int
    cycle: int
    rul: float | None
    status: str
    alerts: list[AlertOut]


class EngineSummary(BaseModel):
    engine_id: int
    cycle: int
    rul: float | None
    status: str


class HistoryPoint(BaseModel):
    cycle: int
    rul: float | None
