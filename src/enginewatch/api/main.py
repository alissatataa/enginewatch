"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session, sessionmaker

from enginewatch import __version__
from enginewatch.config import Settings, get_settings
from enginewatch.db import Alert, Base, Prediction, create_db_engine
from enginewatch.ml.predictor import MissingSensorsError, RulPredictor
from enginewatch.schemas import AlertOut, EngineSummary, HistoryPoint, IngestOut, ReadingIn
from enginewatch.services.alerts import status_for
from enginewatch.services.bus import EventBus
from enginewatch.services.ingest import DuplicateReadingError, ingest_reading

log = logging.getLogger("enginewatch")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def load_predictor(path: str) -> RulPredictor | None:
    p = Path(path)
    if not p.exists():
        log.warning("No RUL model at %s - running without predictions. Run `enginewatch train`.", p)
        return None
    return RulPredictor.load(p)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    bus = EventBus()
    db_engine = create_db_engine(settings.database_url)
    session_factory = sessionmaker(db_engine, expire_on_commit=False)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        Base.metadata.create_all(db_engine)
        bus.bind_loop(asyncio.get_running_loop())
        yield
        db_engine.dispose()

    app = FastAPI(
        title="EngineWatch",
        version=__version__,
        description="Streaming RUL inference and alerting for turbofan engine fleets.",
        lifespan=lifespan,
    )
    app.state.predictor = load_predictor(settings.rul_model_path)

    def get_session() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    def status(rul: float | None) -> str:
        return status_for(rul, settings.warning_rul, settings.critical_rul)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "model_loaded": app.state.predictor is not None}

    @app.post("/readings", response_model=IngestOut, status_code=201, tags=["ingest"])
    def post_reading(payload: ReadingIn, session: Session = Depends(get_session)) -> IngestOut:
        try:
            prediction, alerts = ingest_reading(session, app.state.predictor, settings, payload)
        except DuplicateReadingError as exc:
            raise HTTPException(409, f"Duplicate reading: {exc}") from exc
        except MissingSensorsError as exc:
            raise HTTPException(422, str(exc)) from exc

        st = status(prediction.rul)
        bus.publish(
            {"type": "reading", "engine_id": prediction.engine_id, "cycle": prediction.cycle,
             "rul": prediction.rul, "status": st}
        )  # fmt: skip
        for a in alerts:
            bus.publish(
                {"type": "alert", "engine_id": a.engine_id, "cycle": a.cycle,
                 "level": a.level, "message": a.message}
            )  # fmt: skip
        return IngestOut(
            engine_id=prediction.engine_id,
            cycle=prediction.cycle,
            rul=prediction.rul,
            status=st,
            alerts=[AlertOut.model_validate(a) for a in alerts],
        )

    @app.get("/engines", response_model=list[EngineSummary], tags=["fleet"])
    def list_engines(session: Session = Depends(get_session)) -> list[EngineSummary]:
        latest = (
            select(Prediction.engine_id, func.max(Prediction.cycle).label("cycle"))
            .group_by(Prediction.engine_id)
            .subquery()
        )
        rows = session.scalars(
            select(Prediction)
            .join(
                latest,
                and_(
                    Prediction.engine_id == latest.c.engine_id, Prediction.cycle == latest.c.cycle
                ),
            )
            .order_by(Prediction.engine_id)
        ).all()
        return [
            EngineSummary(engine_id=r.engine_id, cycle=r.cycle, rul=r.rul, status=status(r.rul))
            for r in rows
        ]

    @app.get("/engines/{engine_id}/history", response_model=list[HistoryPoint], tags=["fleet"])
    def engine_history(
        engine_id: int,
        limit: int = Query(500, ge=1, le=5000),
        session: Session = Depends(get_session),
    ) -> list[HistoryPoint]:
        rows = session.execute(
            select(Prediction.cycle, Prediction.rul)
            .where(Prediction.engine_id == engine_id)
            .order_by(Prediction.cycle.desc())
            .limit(limit)
        ).all()
        if not rows:
            raise HTTPException(404, f"Unknown engine {engine_id}")
        return [HistoryPoint(cycle=c, rul=r) for c, r in reversed(rows)]

    @app.get("/alerts", response_model=list[AlertOut], tags=["fleet"])
    def list_alerts(
        engine_id: int | None = None,
        limit: int = Query(50, ge=1, le=500),
        session: Session = Depends(get_session),
    ) -> list[Alert]:
        q = select(Alert).order_by(Alert.id.desc()).limit(limit)
        if engine_id is not None:
            q = q.where(Alert.engine_id == engine_id)
        return list(session.scalars(q).all())

    @app.get("/stream", tags=["live"])
    async def stream(request: Request) -> StreamingResponse:
        async def gen():
            async for event in bus.subscribe():
                if await request.is_disconnected():
                    break
                yield ": keepalive\n\n" if event is None else f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
