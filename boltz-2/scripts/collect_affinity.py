#!/usr/bin/env python3
"""Collect Boltz-2 affinity (and top-model confidence) results into one CSV.

Searches a Boltz output folder for every ``affinity_<ligand>.json`` and writes
one row per ligand with:

* the ensemble values ``affinity_probability_binary`` and ``affinity_pred_value``.
  The ensemble is the mean of models 1 and 2; if the run used
  ``--affinity_mw_correction``, the correction is applied ONLY to the ensemble
  ``affinity_pred_value``.
* the individual model values (``..._1`` and ``..._2``),
* IC50 (μM), pIC50 and an approximate ΔG (kcal/mol) derived from the ensemble
  ``affinity_pred_value``,
* confidence metrics of the top-ranked structure (``model_0``).

Rows are sorted by ``affinity_probability_binary`` (highest first).

Usage:
    python collect_affinity.py                      # auto-detects a folder named "resultados*" / "results*"
    python collect_affinity.py resultados_screen    # explicit results folder
    python collect_affinity.py resultados_screen --decimal-comma   # ';' separators, ',' decimals
"""

import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = [
    "ligand",
    "affinity_probability_binary", "affinity_probability_binary1", "affinity_probability_binary2",
    "affinity_pred_value", "affinity_pred_value1", "affinity_pred_value2",
    "IC50_uM", "pIC50", "dG_kcal_mol_approx",
    "confidence_score", "ligand_iptm", "iptm", "complex_plddt",
    "affinity_json",
]


def find_results_dir(base: Path) -> Path:
    """Find the single results folder in `base` (name contains 'resultados' or 'results')."""
    candidates = sorted(
        p for p in base.iterdir()
        if p.is_dir() and ("resultados" in p.name.lower() or "results" in p.name.lower())
    )
    if not candidates:
        sys.exit(f"No folder containing 'resultados' or 'results' found in {base}. "
                 "Pass the results folder as an argument.")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        sys.exit(f"More than one results folder found ({names}). Pass the one you want as an argument.")
    return candidates[0]


def collect(results_dir: Path) -> list:
    rows = []
    for aff_path in sorted(results_dir.rglob("affinity_*.json")):
        if "predictions" not in aff_path.parts:
            continue  # ignore anything outside predictions/ (e.g. processed/)
        ligand = aff_path.stem[len("affinity_"):]
        try:
            aff = json.loads(aff_path.read_text())
        except (OSError, json.JSONDecodeError) as err:
            print(f"  could not read {aff_path}: {err}")
            continue

        row = {"ligand": ligand, "affinity_json": str(aff_path.relative_to(results_dir))}
        for key in COLUMNS[1:7]:
            row[key] = aff.get(key)

        y = aff.get("affinity_pred_value")  # log10(IC50 in μM), ensemble
        if y is not None:
            row["IC50_uM"] = 10 ** y
            row["pIC50"] = 6 - y
            row["dG_kcal_mol_approx"] = -1.364 * (6 - y)  # ΔG ≈ -RT ln(10) · pIC50 at 298 K

        conf_path = aff_path.parent / f"confidence_{ligand}_model_0.json"
        if conf_path.exists():
            conf = json.loads(conf_path.read_text())
            for key in ("confidence_score", "ligand_iptm", "iptm", "complex_plddt"):
                row[key] = conf.get(key)
        rows.append(row)

    rows.sort(key=lambda r: (r.get("affinity_probability_binary") is None,
                             -(r.get("affinity_probability_binary") or 0)))
    return rows


def fmt(value, decimal_comma: bool) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        text = f"{value:.6g}"
        return text.replace(".", ",") if decimal_comma else text
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results_dir", nargs="?", help="Boltz output folder (the --out_dir used)")
    parser.add_argument("-o", "--output", default="affinity_results.csv", help="output CSV file")
    parser.add_argument("--decimal-comma", action="store_true",
                        help="write ';' separators and ',' decimals (for spreadsheets in pt-BR and similar locales)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else find_results_dir(Path.cwd())
    if not results_dir.is_dir():
        sys.exit(f"Not a folder: {results_dir}")

    rows = collect(results_dir)
    if not rows:
        sys.exit(f"No affinity_*.json files found under {results_dir}/.../predictions/.")

    delimiter = ";" if args.decimal_comma else ","
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([fmt(row.get(c), args.decimal_comma) for c in COLUMNS])

    print(f"Wrote {len(rows)} ligand(s) from {results_dir}/ to {args.output}")


if __name__ == "__main__":
    main()
