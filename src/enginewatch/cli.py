"""Command-line interface: ``enginewatch {train,evaluate,serve,simulate}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from enginewatch.config import get_settings
from enginewatch.data.cmapss import FD001_SENSORS, generate_synthetic, load_cmapss


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, help="Folder with train_/test_/RUL_ C-MAPSS files")
    p.add_argument("--subset", default="FD001")
    p.add_argument("--synthetic", action="store_true", help="Use generated data instead")


def _train_frame(args: argparse.Namespace) -> pd.DataFrame:
    if args.synthetic:
        return generate_synthetic(n_units=60, seed=0)
    if args.data_dir is None:
        sys.exit("error: pass --data-dir or --synthetic")
    return load_cmapss(args.data_dir, args.subset)[0]


def cmd_train(args: argparse.Namespace) -> None:
    from enginewatch.ml.evaluate import evaluate_on_test
    from enginewatch.ml.predictor import train_model

    out = args.out or Path(get_settings().rul_model_path)
    predictor, metrics = train_model(_train_frame(args), FD001_SENSORS, window=args.window)
    predictor.save(out)
    print(f"saved model -> {out}")
    print(f"held-out-engine RMSE: {metrics['val_rmse']:.2f} cycles")
    if args.data_dir and not args.synthetic:
        _, test, rul = load_cmapss(args.data_dir, args.subset)
        res = evaluate_on_test(predictor, test, rul)
        print(f"official test set: RMSE={res['rmse']:.2f}  NASA score={res['nasa_score']:.0f}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "enginewatch.api.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_simulate(args: argparse.Namespace) -> None:
    from enginewatch.simulator import simulate

    df = _train_frame(args)
    units = sorted(df["unit"].unique())[: args.units]
    df = df[df["unit"].isin(units)]
    print(f"replaying {len(units)} engines ({len(df)} readings) at {args.rate}/s -> {args.api}")
    sent = simulate(args.api, df, rate=args.rate)
    print(f"done, {sent} readings sent")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="enginewatch", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train the baseline RUL model")
    _add_data_args(t)
    t.add_argument("--out", type=Path)
    t.add_argument("--window", type=int, default=5)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("serve", help="Run the API + dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    m = sub.add_parser("simulate", help="Replay engines against a running API")
    _add_data_args(m)
    m.add_argument("--api", default="http://127.0.0.1:8000")
    m.add_argument("--units", type=int, default=5)
    m.add_argument("--rate", type=float, default=20.0, help="readings per second (0 = max speed)")
    m.set_defaults(func=cmd_simulate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
