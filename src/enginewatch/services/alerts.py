"""Alert rules. Pure functions so they are trivial to unit test."""

from __future__ import annotations

SEVERITY = {"unknown": 0, "ok": 0, "warning": 1, "critical": 2}


def status_for(rul: float | None, warning: float, critical: float) -> str:
    if rul is None:
        return "unknown"
    if rul <= critical:
        return "critical"
    if rul <= warning:
        return "warning"
    return "ok"


def evaluate_alert(
    highest_alerted: str | None, rul: float | None, warning: float, critical: float
) -> tuple[str, str] | None:
    """Return ``(level, message)`` only if severity exceeds what was already alerted.

    Latching on the highest level raised so far (rather than comparing to the previous
    prediction) makes the rule robust to noisy RUL estimates that jitter around a threshold.
    """
    new = status_for(rul, warning, critical)
    if SEVERITY[new] <= SEVERITY.get(highest_alerted or "ok", 0):
        return None
    threshold = critical if new == "critical" else warning
    return new, f"Estimated RUL is {rul:.0f} cycles (<= {threshold:.0f})"
