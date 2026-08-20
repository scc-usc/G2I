#!/usr/bin/env python
"""
Aggregate the JSON files written by ``run.py`` into Table 3 of the paper.

For every (synthetic family, n_nodes, total_edges) configuration and every
explanation method (CF-Greedy, CF, CF^2) this reports:

    AUCC(Rand.) AUCC(Freq.) AUCC(DNF)   Cov.(DNF)   Time(s)

which are exactly the columns of Table 3. AUCC is the area under the coverage
curve computed by ``src.utils.calculate_auccc``; Cov.(DNF) is the final
coverage of the greedy DNF policy as a fraction of the target nodes; Time is
the wall-clock explanation-generation time.

Usage (from the repository root):
    python scripts/make_table3.py
    python scripts/make_table3.py --results_dir results --csv table3.csv
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import calculate_auccc_from_results  # noqa: E402

FAMILIES = [
    ("synthetic_neighbor_feature", "Neighbor-Feature"),
    ("synthetic_neighbor_only", "Neighbor-Only"),
]
METHOD_ORDER = ["cf", "cff", "cf-greedy"]
METHOD_LABEL = {"cf": "CF", "cff": "CF^2", "cf-greedy": "Ours-Greedy"}


def parse_config(path):
    """Pull (n_nodes, m_attrs, total_edges) out of a result filename."""
    name = os.path.basename(path)
    m = re.search(r"node(\d+)_attr(\d+)_edge(\d+)", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def summarize_file(path):
    """Return {method: {metric: (mean, std)}} for one result file."""
    with open(path, "r") as f:
        results = json.load(f)

    runs = results.get("runs", [])
    if not runs:
        return {}

    auccc = calculate_auccc_from_results(results)

    out = {}
    for method in runs[0].get("methods", {}):
        coverage, times = [], []
        for run in runs:
            md = run["methods"].get(method)
            if md is None:
                continue
            n_target = run.get("num_target_nodes", 0)
            times.append(md["running_time"])
            if "greedy" in md and n_target > 0:
                coverage.append(md["greedy"]["final_coverage"] / n_target)

        def agg(values):
            if not values:
                return None
            arr = np.asarray(values, dtype=float)
            return float(arr.mean()), float(arr.std())

        entry = {
            "rand": agg(auccc.get(method, {}).get("random", [])),
            "freq": agg(auccc.get(method, {}).get("frequency", [])),
            "dnf": agg(auccc.get(method, {}).get("greedy", [])),
            "coverage": agg(coverage),
            "time": agg(times),
        }
        out[method] = entry
    return out


def fmt(pair, pct=False, digits=3):
    if pair is None:
        return "/"
    mean, std = pair
    if pct:
        return f"{mean * 100:.2f}%"
    return f"{mean:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing the synthetic_* result folders")
    parser.add_argument("--csv", default=None,
                        help="Optional path to also write the table as CSV")
    parser.add_argument("--show_std", action="store_true",
                        help="Print mean +/- std instead of the mean alone "
                             "(the paper reports means only)")
    args = parser.parse_args()

    rows = []
    for folder, family_label in FAMILIES:
        pattern = os.path.join(args.results_dir, folder, "*.json")
        for path in sorted(glob.glob(pattern)):
            cfg = parse_config(path)
            if cfg is None:
                print(f"  (skipping unrecognized filename: {path})")
                continue
            n_nodes, m_attrs, total_edges = cfg
            summary = summarize_file(path)
            if not summary:
                print(f"  (skipping empty result file: {path})")
                continue
            label = f"N{n_nodes}-E{total_edges}-D{m_attrs}"
            for method in METHOD_ORDER:
                if method not in summary:
                    continue
                e = summary[method]
                rows.append({
                    "family": family_label,
                    "dataset": label,
                    "n_nodes": n_nodes,
                    "total_edges": total_edges,
                    "m_attrs": m_attrs,
                    "method": METHOD_LABEL[method],
                    "auccc_rand": e["rand"],
                    "auccc_freq": e["freq"],
                    "auccc_dnf": e["dnf"],
                    "coverage_dnf": e["coverage"],
                    "time_s": e["time"],
                })

    if not rows:
        print(f"No result files found under {args.results_dir}/synthetic_*/.")
        print("Run `bash scripts/run_table3.sh` first.")
        return 1

    def cell(pair, pct=False, digits=3):
        if pair is None:
            return "/"
        if args.show_std:
            mean, std = pair
            if pct:
                return f"{mean * 100:.2f}%+-{std * 100:.2f}%"
            return f"{mean:.{digits}f}+-{std:.{digits}f}"
        return fmt(pair, pct=pct, digits=digits)

    header = (f'{"Family":<17} {"Dataset":<17} {"Method":<12} '
              f'{"Rand.":<16} {"Freq.":<16} {"DNF":<16} '
              f'{"Cov.(DNF)":<18} {"Time(s)":<16}')
    print()
    print("Table 3 - Intervention evaluation (synthetic configurations)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    last_key = None
    for r in rows:
        key = (r["family"], r["dataset"])
        if last_key is not None and key != last_key:
            print()
        family = r["family"] if key != last_key else ""
        dataset = r["dataset"] if key != last_key else ""
        print(f'{family:<17} {dataset:<17} {r["method"]:<12} '
              f'{cell(r["auccc_rand"]):<16} {cell(r["auccc_freq"]):<16} '
              f'{cell(r["auccc_dnf"]):<16} {cell(r["coverage_dnf"], pct=True):<18} '
              f'{cell(r["time_s"], digits=3):<16}')
        last_key = key
    print("=" * len(header))

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["family", "dataset", "n_nodes", "total_edges", "m_attrs",
                             "method", "auccc_rand", "auccc_freq", "auccc_dnf",
                             "coverage_dnf", "time_s"])
            for r in rows:
                writer.writerow([
                    r["family"], r["dataset"], r["n_nodes"], r["total_edges"],
                    r["m_attrs"], r["method"],
                    "" if r["auccc_rand"] is None else r["auccc_rand"][0],
                    "" if r["auccc_freq"] is None else r["auccc_freq"][0],
                    "" if r["auccc_dnf"] is None else r["auccc_dnf"][0],
                    "" if r["coverage_dnf"] is None else r["coverage_dnf"][0],
                    "" if r["time_s"] is None else r["time_s"][0],
                ])
        print(f"\nCSV written to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
