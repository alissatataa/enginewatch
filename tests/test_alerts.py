from enginewatch.services.alerts import evaluate_alert, status_for

W, C = 60.0, 30.0


def test_status_thresholds():
    assert status_for(None, W, C) == "unknown"
    assert status_for(100, W, C) == "ok"
    assert status_for(60, W, C) == "warning"
    assert status_for(30, W, C) == "critical"


def test_alert_escalates_once_per_level():
    assert evaluate_alert(None, 80, W, C) is None
    level, msg = evaluate_alert(None, 55, W, C)
    assert level == "warning" and "55" in msg
    assert evaluate_alert("warning", 50, W, C) is None  # already alerted at this level
    assert evaluate_alert("warning", 20, W, C)[0] == "critical"


def test_noisy_rul_does_not_realert():
    # RUL jitters back above the critical threshold and dips again: still only one critical
    assert evaluate_alert("critical", 25, W, C) is None
    assert evaluate_alert("critical", 40, W, C) is None
    assert evaluate_alert("critical", 28, W, C) is None


def test_first_reading_already_critical_alerts():
    assert evaluate_alert(None, 10, W, C)[0] == "critical"


def test_no_model_never_alerts():
    assert evaluate_alert(None, None, W, C) is None
