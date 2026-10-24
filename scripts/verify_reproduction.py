"""Compare a new full experiment with the preserved reference results.

Run from the repository root:
python scripts/verify_reproduction.py --reference results --candidate rerun
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(reference, candidate):
    report = {"reference": str(reference), "candidate": str(candidate), "csv": {}, "arrays": {}}
    ok = True
    for name in ("all_metrics.csv", "clean_metrics.csv", "primary_summary.csv",
                 "table1.csv", "table2.csv", "error_association.csv", "severity_trends.csv"):
        left, right = (pd.read_csv(folder / name) for folder in (reference, candidate))
        if list(left.columns) != list(right.columns) or left.shape != right.shape:
            raise ValueError(f"Schema or shape differs: {name}")
        numeric = list(left.select_dtypes(include="number").columns)
        nonnumeric = [column for column in left if column not in numeric]
        labels_equal = left[nonnumeric].equals(right[nonnumeric])
        x, y = left[numeric].to_numpy(float), right[numeric].to_numpy(float)
        values_match = bool(np.allclose(x, y, atol=5e-8, rtol=0, equal_nan=True))
        finite = np.isfinite(x) & np.isfinite(y)
        maximum = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.0
        report["csv"][name] = {"rows": len(left), "labels_equal": labels_equal,
                                "maximum_absolute_difference": maximum,
                                "within_absolute_tolerance_5e_8": values_match}
        ok = ok and labels_equal and values_match
    with np.load(reference / "case_audit.npz", allow_pickle=False) as left, np.load(candidate / "case_audit.npz", allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            raise ValueError("Case audit array names differ")
        exact_decisions, maximum = True, 0.0
        for name in left.files:
            x, y = left[name], right[name]
            if x.shape != y.shape:
                raise ValueError(f"Array shape differs: {name}")
            exact = name.endswith(("_flip_frequency", "_row_ids", "_clean_error")) or name == "target"
            if exact:
                matched = bool(np.array_equal(x, y))
                exact_decisions = exact_decisions and matched
            else:
                matched = bool(np.allclose(x, y, atol=5e-8, rtol=0, equal_nan=True))
                maximum = max(maximum, float(np.max(np.abs(x - y))))
            report["arrays"][name] = {"exact_comparison": exact, "match": matched}
            ok = ok and matched
        report["case_array_count"] = len(left.files)
        report["decision_frequencies_row_ids_labels_and_errors_identical"] = exact_decisions
        report["maximum_probability_or_margin_difference"] = maximum
    reference_meta, candidate_meta = (json.loads((folder / "run_metadata.json").read_text()) for folder in (reference, candidate))
    metadata_keys = ("input_csv_sha256", "source_sha256", "rows", "seed", "repetitions",
                     "bootstrap_draws", "severity_levels", "diagnostic_random_only", "execution_complete")
    report["protocol_metadata_equal"] = all(reference_meta[key] == candidate_meta[key] for key in metadata_keys)
    report["candidate_environment"] = {key: candidate_meta[key] for key in ("python", "platform", "numpy", "pandas", "scipy", "scikit_learn", "matplotlib", "elapsed_seconds")}
    report["candidate_source_hash_matches_repository"] = candidate_meta["source_sha256"] == hashlib.sha256(Path("source/research.py").read_bytes()).hexdigest()
    ok = ok and report["protocol_metadata_equal"] and report["candidate_source_hash_matches_repository"]
    report["passed"] = bool(ok)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("results"))
    parser.add_argument("--candidate", type=Path, default=Path("rerun"))
    parser.add_argument("--report", type=Path, default=Path("reproduction_comparison.json"))
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("arrays", "csv")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
