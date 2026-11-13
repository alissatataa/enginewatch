"""The ingest pipeline: persist reading -> predict RUL -> evaluate alert rules."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from enginewatch.config import Settings
from enginewatch.db import Alert, Prediction, Reading
from enginewatch.ml.predictor import RulPredictor
from enginewatch.schemas import ReadingIn
from enginewatch.services.alerts import SEVERITY, evaluate_alert


class DuplicateReadingError(Exception):
    pass


def ingest_reading(
    session: Session,
    predictor: RulPredictor | None,
    settings: Settings,
    payload: ReadingIn,
) -> tuple[Prediction, list[Alert]]:
    exists = session.scalar(
        select(Reading.id).where(
            Reading.engine_id == payload.engine_id, Reading.cycle == payload.cycle
        )
    )
    if exists is not None:
        raise DuplicateReadingError(f"engine {payload.engine_id} cycle {payload.cycle}")

    session.add(
        Reading(
            engine_id=payload.engine_id,
            cycle=payload.cycle,
            settings=payload.settings,
            sensors=payload.sensors,
        )
    )
    session.flush()

    rul: float | None = None
    if predictor is not None:
        rows = session.execute(
            select(Reading.cycle, Reading.sensors)
            .where(Reading.engine_id == payload.engine_id, Reading.cycle <= payload.cycle)
            .order_by(Reading.cycle.desc())
            .limit(predictor.window)
        ).all()
        rows.reverse()
        rul = predictor.predict_one([r.sensors for r in rows], [r.cycle for r in rows])

    raised = session.scalars(select(Alert.level).where(Alert.engine_id == payload.engine_id)).all()
    highest = max(raised, key=lambda lvl: SEVERITY[lvl], default=None)

    prediction = Prediction(engine_id=payload.engine_id, cycle=payload.cycle, rul=rul)
    session.add(prediction)

    alerts: list[Alert] = []
    triggered = evaluate_alert(highest, rul, settings.warning_rul, settings.critical_rul)
    if triggered:
        level, message = triggered
        alert = Alert(
            engine_id=payload.engine_id, cycle=payload.cycle, level=level, message=message
        )
        session.add(alert)
        alerts.append(alert)

    session.commit()
    return prediction, alerts
