"""
run_multi_seed.py
=================

Driver for running the five RWSD experiments across many seeds and
saving all outputs to disk.

Usage
-----

Run all 5 experiments with seeds 0..19 (default) into ./rwsd_results::

    python run_multi_seed.py

Pick a subset of experiments and seeds::

    python run_multi_seed.py \\
        --experiments exp1_imbalanced_dispersion exp3_heterogeneous_scales \\
        --seeds 0 1 2 3 4 \\
        --output-dir ./my_run

Re-run from scratch (overwrite existing per-seed JSONs)::

    python run_multi_seed.py --no-skip-existing

Aggregate-only (no re-clustering, just rebuild summary CSVs from
existing per-seed JSONs)::

    python run_multi_seed.py --aggregate-only

Output layout
-------------

    {output_dir}/
        per_seed/
            {experiment}/
                seed_{seed:03d}.json    # full per-run record
        summary_metrics.csv             # long format (experiment, seed, K, sil, red, dbi)
        summary_selections.csv          # one row per (experiment, seed)
        summary_aggregate.csv           # one row per experiment with hit rates
        run_log.txt                     # log of all runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import rwsd_core as core
from rwsd_experiments import EXPERIMENTS


# ============================================================
# Single (experiment, seed) run
# ============================================================

def run_one(experiment_name, seed, n_jobs=-1, reg=0.12, verbose=True):
    """Run one experiment at one seed and return a JSON-serializable record."""
    spec = EXPERIMENTS[experiment_name]
    Ks = spec["Ks"]
    method = spec["method"]
    true_K = spec["true_K"]

    t0 = time.time()
    distributions, labels_true = spec["call"](seed=seed)

    # Build a CONFIG with the requested seed; this drives the clustering RNG.
    config = core.SimpleNamespace(seed=seed, reg=reg, n_jobs=n_jobs, no_plot=True, output_dir=None)

    out = core.validate_by_method(
        distributions=distributions,
        Ks=Ks,
        method=method,
        config=config,
        plot_validation=False,
        use_pam_warm_start=True,
    )

    runtime = time.time() - t0

    # Coerce numpy types -> python primitives so json.dump works directly.
    def _safe_float(x):
        try:
            x = float(x)
            return x if np.isfinite(x) else None
        except Exception:
            return None

    record = {
        "experiment": experiment_name,
        "seed": int(seed),
        "method": method,
        "Ks": list(Ks),
        "true_K": int(true_K),
        "has_noise": bool(spec["has_noise"]),
        "n_distributions": int(len(distributions)),
        "metrics": {
            "silhouette": {str(K): _safe_float(out["sil_vals"][K]) for K in Ks},
            "ReD":        {str(K): _safe_float(out["red_vals"][K]) for K in Ks},
            "DBI":        {str(K): _safe_float(out["dbi_vals"][K]) for K in Ks},
        },
        "selected_K": {
            "silhouette": (int(out["K_star_sil"]) if out["K_star_sil"] is not None else None),
            "ReD":        (int(out["K_star_red"]) if out["K_star_red"] is not None else None),
            "DBI":        (int(out["K_star_dbi"]) if out["K_star_dbi"] is not None else None),
        },
        "labels_true": [int(x) for x in np.asarray(labels_true).tolist()],
        "labels_perK": {
            str(K): [int(x) for x in np.asarray(out["labels_perK"][K]).tolist()]
            for K in Ks
        },
        "runtime_seconds": float(runtime),
    }

    if verbose:
        print(
            f"[done] {experiment_name} seed={seed} "
            f"K_sil={record['selected_K']['silhouette']} "
            f"K_red={record['selected_K']['ReD']} "
            f"K_dbi={record['selected_K']['DBI']} "
            f"true_K={true_K} runtime={runtime:.1f}s"
        )
    return record


# ============================================================
# Aggregation: build the three summary CSVs from per-seed JSONs
# ============================================================

def aggregate_results(output_dir):
    """Walk per_seed/ and build summary_metrics.csv, summary_selections.csv, summary_aggregate.csv."""
    output_dir = Path(output_dir)
    per_seed_dir = output_dir / "per_seed"
    if not per_seed_dir.exists():
        print(f"[aggregate] No per_seed directory at {per_seed_dir}; nothing to do.")
        return

    # Long-format metrics: one row per (experiment, seed, K).
    metrics_rows = []
    selections_rows = []

    for exp_dir in sorted(per_seed_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for js in sorted(exp_dir.glob("seed_*.json")):
            with open(js) as f:
                rec = json.load(f)
            exp = rec["experiment"]
            seed = rec["seed"]
            true_K = rec["true_K"]

            for K in rec["Ks"]:
                metrics_rows.append({
                    "experiment": exp,
                    "seed": seed,
                    "K": K,
                    "silhouette": rec["metrics"]["silhouette"].get(str(K)),
                    "ReD":        rec["metrics"]["ReD"].get(str(K)),
                    "DBI":        rec["metrics"]["DBI"].get(str(K)),
                })

            sel = rec["selected_K"]
            selections_rows.append({
                "experiment": exp,
                "seed": seed,
                "true_K": true_K,
                "method": rec["method"],
                "K_silhouette": sel["silhouette"],
                "K_ReD":        sel["ReD"],
                "K_DBI":        sel["DBI"],
                "silhouette_correct": int(sel["silhouette"] == true_K) if sel["silhouette"] is not None else None,
                "ReD_correct":        int(sel["ReD"]        == true_K) if sel["ReD"]        is not None else None,
                "DBI_correct":        int(sel["DBI"]        == true_K) if sel["DBI"]        is not None else None,
                "runtime_seconds": rec["runtime_seconds"],
            })

    if not selections_rows:
        print("[aggregate] No per-seed files found; nothing to aggregate.")
        return

    # Use pandas if available; otherwise hand-write CSVs.
    try:
        import pandas as pd
        df_metrics = pd.DataFrame(metrics_rows)
        df_sel = pd.DataFrame(selections_rows)

        df_metrics.to_csv(output_dir / "summary_metrics.csv", index=False)
        df_sel.to_csv(output_dir / "summary_selections.csv", index=False)

        agg_rows = []
        for exp, grp in df_sel.groupby("experiment"):
            row = {"experiment": exp, "n_seeds": len(grp), "true_K": int(grp["true_K"].iloc[0])}
            for crit in ["silhouette", "ReD", "DBI"]:
                col = f"{crit}_correct"
                ok = grp[col].dropna()
                row[f"{crit}_hit_rate"] = float(ok.mean()) if len(ok) else None
                row[f"{crit}_n_valid"] = int(len(ok))
                kc = grp[f"K_{crit}"].dropna()
                if len(kc):
                    row[f"{crit}_K_mode"] = int(kc.mode().iloc[0])
                    row[f"{crit}_K_mean"] = float(kc.mean())
                    row[f"{crit}_K_std"] = float(kc.std()) if len(kc) > 1 else 0.0
                else:
                    row[f"{crit}_K_mode"] = None
                    row[f"{crit}_K_mean"] = None
                    row[f"{crit}_K_std"] = None
            agg_rows.append(row)
        df_agg = pd.DataFrame(agg_rows)
        df_agg.to_csv(output_dir / "summary_aggregate.csv", index=False)

        print("\n[aggregate] Wrote:")
        print(f"  - {output_dir / 'summary_metrics.csv'}    ({len(df_metrics)} rows)")
        print(f"  - {output_dir / 'summary_selections.csv'} ({len(df_sel)} rows)")
        print(f"  - {output_dir / 'summary_aggregate.csv'}  ({len(df_agg)} rows)")
        print("\n=== Aggregate hit rates ===")
        print(df_agg.to_string(index=False))

    except ImportError:
        # Fallback: write CSVs without pandas.
        import csv
        with open(output_dir / "summary_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            w.writeheader(); w.writerows(metrics_rows)
        with open(output_dir / "summary_selections.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(selections_rows[0].keys()))
            w.writeheader(); w.writerows(selections_rows)

        agg = {}
        for r in selections_rows:
            d = agg.setdefault(r["experiment"], {"n": 0, "true_K": r["true_K"],
                                                  "sil_ok": [], "red_ok": [], "dbi_ok": [],
                                                  "K_sil": [], "K_red": [], "K_dbi": []})
            d["n"] += 1
            for k_short, k_long in [("sil", "silhouette"), ("red", "ReD"), ("dbi", "DBI")]:
                if r[f"{k_long}_correct"] is not None:
                    d[f"{k_short}_ok"].append(r[f"{k_long}_correct"])
                if r[f"K_{k_long}"] is not None:
                    d[f"K_{k_short}"].append(r[f"K_{k_long}"])

        agg_rows = []
        for exp, d in agg.items():
            row = {"experiment": exp, "n_seeds": d["n"], "true_K": d["true_K"]}
            for k_short, k_long in [("sil", "silhouette"), ("red", "ReD"), ("dbi", "DBI")]:
                row[f"{k_long}_hit_rate"] = (sum(d[f"{k_short}_ok"]) / len(d[f"{k_short}_ok"])
                                              if d[f"{k_short}_ok"] else None)
                row[f"{k_long}_n_valid"] = len(d[f"{k_short}_ok"])
            agg_rows.append(row)

        with open(output_dir / "summary_aggregate.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader(); w.writerows(agg_rows)

        print(f"[aggregate] Wrote summary CSVs (pandas not available; reduced columns).")


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS.keys()),
                   help="Experiment names to run (default: all 5).")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(20)),
                   help="Seed values to run (default: 0..19).")
    p.add_argument("--output-dir", default="./rwsd_results",
                   help="Where to write outputs (default: ./rwsd_results).")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallel workers for joblib (default: -1 = all cores).")
    p.add_argument("--reg", type=float, default=0.12,
                   help="Sinkhorn regularization (default: 0.12).")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-run even if a per-seed JSON already exists.")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip clustering; just rebuild summary CSVs from existing per-seed JSONs.")
    args = p.parse_args()

    # Validate experiment names early.
    bad = [e for e in args.experiments if e not in EXPERIMENTS]
    if bad:
        print(f"Unknown experiments: {bad}", file=sys.stderr)
        print(f"Available: {list(EXPERIMENTS.keys())}", file=sys.stderr)
        sys.exit(2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_seed").mkdir(exist_ok=True)
    log_path = output_dir / "run_log.txt"

    if args.aggregate_only:
        aggregate_results(output_dir)
        return

    total = len(args.experiments) * len(args.seeds)
    done = 0
    skipped = 0
    failed = 0

    with open(log_path, "a") as logf:
        logf.write(f"\n========== Run started {time.strftime('%Y-%m-%d %H:%M:%S')} ==========\n")
        logf.write(f"experiments: {args.experiments}\n")
        logf.write(f"seeds: {args.seeds}\n")

        for exp_name in args.experiments:
            exp_dir = output_dir / "per_seed" / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            for seed in args.seeds:
                done += 1
                out_path = exp_dir / f"seed_{seed:03d}.json"
                tag = f"[{done}/{total}] {exp_name} seed={seed}"

                if out_path.exists() and not args.no_skip_existing:
                    skipped += 1
                    msg = f"{tag}  SKIP (exists)"
                    print(msg)
                    logf.write(msg + "\n")
                    logf.flush()
                    continue

                msg_start = f"{tag}  START at {time.strftime('%H:%M:%S')}"
                print(msg_start)
                logf.write(msg_start + "\n"); logf.flush()

                try:
                    record = run_one(
                        exp_name, seed,
                        n_jobs=args.n_jobs, reg=args.reg, verbose=True,
                    )
                    with open(out_path, "w") as f:
                        json.dump(record, f, indent=2)
                    msg_end = f"{tag}  OK    runtime={record['runtime_seconds']:.1f}s"
                    print(msg_end)
                    logf.write(msg_end + "\n"); logf.flush()
                except Exception:
                    failed += 1
                    tb = traceback.format_exc()
                    msg_err = f"{tag}  FAIL\n{tb}"
                    print(msg_err)
                    logf.write(msg_err + "\n"); logf.flush()

        summary = (
            f"\n========== Run finished {time.strftime('%Y-%m-%d %H:%M:%S')} ==========\n"
            f"total={total} skipped={skipped} failed={failed}\n"
        )
        print(summary)
        logf.write(summary)

    # Always re-aggregate at the end so the CSVs reflect what's on disk.
    aggregate_results(output_dir)


if __name__ == "__main__":
    main()
