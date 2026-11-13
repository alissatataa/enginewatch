"""Replays a C-MAPSS-shaped dataframe against the API as if engines were running live."""

from __future__ import annotations

import time

import httpx
import pandas as pd

from enginewatch.data.cmapss import row_to_payload


def simulate(api_url: str, df: pd.DataFrame, rate: float = 20.0, timeout: float = 10.0) -> int:
    """POST readings in lockstep (cycle 1 of every engine, then 2, ...). Returns count sent."""
    df = df.sort_values(["cycle", "unit"])
    delay = 1.0 / rate if rate > 0 else 0.0
    sent = 0
    with httpx.Client(base_url=api_url, timeout=timeout) as client:
        for _, row in df.iterrows():
            resp = client.post("/readings", json=row_to_payload(row))
            if resp.status_code != 409:  # 409 = already ingested (re-running the simulator)
                resp.raise_for_status()
                sent += 1
            if sent and sent % 100 == 0:
                print(f"sent {sent} readings...", flush=True)
            if delay:
                time.sleep(delay)
    return sent
