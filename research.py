#!/usr/bin/env python3
"""Paired input-corruption audit. Python 3.11/3.12; see README.
The full experiment has NOT been run by the authoring assistant.
No data, performance values or publication readiness are simulated.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import platform
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
TITLE = ("Data Quality-Aware AI Decision Intelligence: Quantifying "
         "Decision Stability Under Missing and Noisy Data")
LEVELS = (0.05, 0.10, 0.20, 0.30)
NAMES = ("Logistic Regression", "Random Forest", "HistGradientBoosting",
         "Constant baseline")
KEY = {n: ("lr", "rf", "hgb", "constant")[i] for i, n in enumerate(NAMES)}
COLORS = dict(zip(NAMES, ("#0072B2", "#D55E00", "#009E73", "#666666")))
COUNT_COLS = {"age", "campaign", "pdays", "previous"}
METRICS = ("accuracy", "balanced_accuracy", "roc_auc", "average_precision",
           "f1", "brier", "ece10", "action_rate", "dfr", "dss",
           "harmful", "beneficial", "mean_probability_shift",
           "mean_absolute_probability_shift", "delta_balanced_accuracy",
           "delta_roc_auc")


def read_data(csv_path: str | None, out: Path):
    if csv_path:
        raw = Path(csv_path).read_bytes()
        provenance = str(Path(csv_path).resolve())
    else:
        with urllib.request.urlopen(URL, timeout=90) as response:
            archive = response.read()
        outer = zipfile.ZipFile(io.BytesIO(archive))
        names = outer.namelist()
        matches = [n for n in names if n.endswith("bank-additional-full.csv")]
        if matches:
            raw = outer.read(matches[0])
        else:
            nested = [n for n in names if n.endswith("bank-additional.zip")]
            if len(nested) != 1:
                raise ValueError("Expected the bank-additional ZIP from UCI.")
            inner = zipfile.ZipFile(io.BytesIO(outer.read(nested[0])))
            match = [n for n in inner.namelist()
                     if n.endswith("bank-additional-full.csv")]
            if len(match) != 1:
                raise ValueError("Could not resolve bank-additional-full.csv.")
            raw = inner.read(match[0])
        provenance = URL
    df = pd.read_csv(io.BytesIO(raw), sep=";")
    if len(df) != 41188 or len(df.columns) != 21:
        raise ValueError("Use bank-additional-full.csv: 41,188 rows, 21 columns.")
    if "duration" not in df or set(df.y.unique()) != {"yes", "no"}:
        raise ValueError("Unexpected dataset schema or target.")
    y = df.pop("y").eq("yes").astype("int8").to_numpy()
    df = df.drop(columns="duration")
    if df.isna().any().any():
        raise ValueError("Unexpected literal missing cells; inspect the source.")
    # 'unknown' is retained as an observed category, not replaced by NaN.
    meta = {"dataset_url": URL, "input_source": provenance,
            "input_csv_sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(df), "raw_predictors_after_duration_removal": len(df.columns),
            "positive_count": int(y.sum()), "positive_prevalence": float(y.mean()),
            "data_license": "CC BY 4.0",
            "dataset_doi": "10.24432/C5K306",
            "duration_removed": True, "unknown_category_retained": True}
    return df, y, meta


def partitions(y, seed, scheme):
    ids = np.arange(len(y))
    if scheme == "random":
        development, test = train_test_split(
            ids, test_size=0.20, random_state=seed, stratify=y)
        train, validation = train_test_split(
            development, test_size=0.25, random_state=seed + 1,
            stratify=y[development])
    else:
        # UCI describes this CSV as ordered by date; do not shuffle.
        n_test = int(np.ceil(0.20 * len(y)))
        n_validation = int(np.ceil(0.25 * (len(y) - n_test)))
        train = ids[:len(y) - n_test - n_validation]
        validation = ids[len(train):len(y) - n_test]
        test = ids[len(y) - n_test:]
    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)
    assert len(train) + len(validation) + len(test) == len(y)
    if any(len(np.unique(y[k])) != 2 for k in (train, validation, test)):
        raise ValueError("Each partition must contain both outcome classes.")
    return train, validation, test


def preprocessor(train):
    numeric = train.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in train if c not in numeric]
    transformer = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore",
                                            sparse_output=False)),
         categorical)], sparse_threshold=0)
    return transformer, numeric, categorical


def fit_models(train, labels, seed, jobs):
    models = {
        NAMES[0]: LogisticRegression(C=1.0, max_iter=3000,
                                    solver="lbfgs", random_state=seed),
        NAMES[1]: RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, max_features="sqrt",
            n_jobs=jobs, random_state=seed),
        NAMES[2]: HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=False, random_state=seed),
        # A fixed 0.5 threshold is used for this negative-control model.
        NAMES[3]: DummyClassifier(strategy="prior"),
    }
    for name, model in models.items():
        print("  fitting", name, flush=True)
        model.fit(train, labels)
    return models


def choose_threshold(y, p):
    fpr, tpr, threshold = roc_curve(y, p, drop_intermediate=False)
    value = (tpr + 1 - fpr) / 2
    eligible = np.flatnonzero(np.isfinite(threshold) &
                             np.isclose(value, np.max(value), atol=1e-12,
                                        rtol=0))
    if not len(eligible):
        return 0.5
    return float(threshold[eligible[np.argmin(
        np.abs(threshold[eligible] - 0.5))]])


def ece10(y, p):
    # Positive-class calibration, ten fixed equal-width bins.
    bins = np.minimum((np.clip(p, 0, 1) * 10).astype(int), 9)
    return float(sum(np.mean(bins == b) *
                     abs(np.mean(y[bins == b]) - np.mean(p[bins == b]))
                     for b in range(10) if np.any(bins == b)))


def metrics(y, p, threshold, reference):
    d = p >= threshold
    d0 = reference >= threshold
    flips = d != d0
    correct0 = d0 == y
    harmful = flips & correct0
    beneficial = flips & ~correct0
    result = {
        "accuracy": accuracy_score(y, d),
        "balanced_accuracy": balanced_accuracy_score(y, d),
        "roc_auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "f1": f1_score(y, d, zero_division=0),
        "brier": brier_score_loss(y, p), "ece10": ece10(y, p),
        "action_rate": np.mean(d), "dfr": np.mean(flips),
        "dss": 1 - np.mean(flips), "harmful": np.mean(harmful),
        "beneficial": np.mean(beneficial),
        "mean_probability_shift": np.mean(p - reference),
        "mean_absolute_probability_shift": np.mean(np.abs(p - reference)),
        "delta_balanced_accuracy": (balanced_accuracy_score(y, d) -
                                   balanced_accuracy_score(y, d0)),
        "delta_roc_auc": roc_auc_score(y, p) - roc_auc_score(y, reference)}
    assert np.isclose(result["dfr"], result["harmful"] + result["beneficial"])
    assert np.isclose(result["accuracy"] - accuracy_score(y, d0),
                      result["beneficial"] - result["harmful"])
    assert np.isclose(result["dss"] + result["dfr"], 1)
    return {k: float(v) for k, v in result.items()}, flips.astype("float32")


def corruption_spec(train, numeric, categorical):
    spec = {}
    for c in numeric:
        values = train[c].to_numpy(dtype=float)
        values = values[values != 999] if c == "pdays" else values
        spec[c] = float(np.std(values, ddof=0))
    for c in categorical:
        spec[c] = np.sort(train[c].unique()).tolist()
    return spec


def corrupt(test, numeric, categorical, spec, rate, kind, seed):
    # Reinitializing at each rate creates nested masks within a repetition.
    # The same corrupted rows are used across all classifiers and thresholds.
    rng = np.random.default_rng(seed)
    changed = test.copy()
    attempted = realized = eligible_total = 0
    for c in test:
        n = len(test)
        mask = rng.random(n) < rate
        original = test[c].to_numpy()
        if kind == "missing":
            eligible = np.ones(n, dtype=bool)
            candidate = np.full(n, np.nan)
        elif c in numeric:
            eligible = original != 999 if c == "pdays" else np.ones(n, bool)
            candidate = original.astype(float) + rng.normal(
                0, 0.25 * spec[c], n)
            if c in COUNT_COLS:
                candidate = np.rint(candidate)
                candidate = np.maximum(candidate, 0 if c != "age" else 1)
            if c == "pdays":
                # Keep sentinel semantics; do not turn non-sentinel days into 999.
                candidate[candidate == 999] = 998
        else:
            vocabulary = np.array(spec[c], dtype=object)
            eligible = np.full(n, len(vocabulary) >= 2)
            # Draw uniformly from training categories other than the original.
            candidate = vocabulary[rng.integers(0, len(vocabulary), n)].copy()
            equal = candidate == original
            if len(vocabulary) >= 2:
                while np.any(equal):
                    candidate[equal] = vocabulary[rng.integers(
                        0, len(vocabulary), np.sum(equal))]
                    equal = candidate == original
        mask &= eligible
        values = original.astype(float if c in numeric else object).copy()
        values[mask] = candidate[mask]
        changed[c] = values
        attempted += int(mask.sum())
        eligible_total += int(eligible.sum())
        realized += (int(mask.sum()) if kind == "missing" else
                     int(np.sum(mask & (candidate != original))))
    denominator = test.shape[0] * test.shape[1]
    return changed, {"attempted_cell_fraction": attempted / denominator,
                     "realized_cell_fraction": realized / denominator,
                     "eligible_cell_fraction": eligible_total / denominator}


def case_bootstrap_ci(values, seed, n_boot):
    # Each row represents one case averaged across corruption repetitions.
    # Conditional on the fitted model, split, and realized corruption draws.
    rng = np.random.default_rng(seed)
    boot = np.array([values[rng.integers(0, len(values), len(values))].mean()
                     for _ in range(n_boot)])
    return tuple(float(x) for x in np.quantile(boot, [0.025, 0.975]))


def holm(p):
    order = np.argsort(p)
    adjusted = np.empty(len(p), float)
    ranked = np.maximum.accumulate(
        np.array(p)[order] * (len(p) - np.arange(len(p))))
    adjusted[order] = np.minimum(ranked, 1)
    return adjusted


def tables_and_figures(records, clean, cases, args, out):
    frame = pd.DataFrame(records)
    frame.to_csv(out / "all_metrics.csv", index=False)
    pd.DataFrame(clean).to_csv(out / "clean_metrics.csv", index=False)
    primary = frame.query("split == 'random' and policy == 'validation'")
    summary = primary.groupby(["model", "mechanism", "rate"])[
        list(METRICS)].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    for name in ("dfr_ci_low", "dfr_ci_high"):
        summary[name] = np.nan
    association = []
    for i, row in summary.iterrows():
        item = cases[("random", row.model, row.mechanism, row.rate)]
        frequency = item["frequency"]
        ci = case_bootstrap_ci(frequency, args.seed + 301, args.bootstrap)
        summary.loc[i, ["dfr_ci_low", "dfr_ci_high"]] = ci
        if row.rate == 0.30 and row.model != NAMES[3]:
            errors = item["clean_error"]
            negative_margin = -item["margin"]
            p = mannwhitneyu(frequency[errors], frequency[~errors],
                             alternative="two-sided", method="asymptotic").pvalue
            association.append({
                "model": row.model, "mechanism": row.mechanism,
                "flip_error_auc": roc_auc_score(errors, frequency),
                "margin_error_auc": roc_auc_score(errors, negative_margin),
                "mann_whitney_p_exploratory": float(p)})
    assoc = pd.DataFrame(association)
    assoc["holm_p_exploratory"] = holm(
        assoc["mann_whitney_p_exploratory"].to_numpy())
    assoc.to_csv(out / "error_association.csv", index=False)
    trends = []
    for (name, kind), g in summary.groupby(["model", "mechanism"]):
        rho = (float(spearmanr(g.rate, g.dfr_mean).statistic)
               if g.dfr_mean.nunique() > 1 else np.nan)
        trends.append({"model": name, "mechanism": kind,
                       "spearman_rho_descriptive": rho,
                       "severity_points": 4})
    pd.DataFrame(trends).to_csv(out / "severity_trends.csv", index=False)
    summary.to_csv(out / "primary_summary.csv", index=False)
    t1 = pd.DataFrame(clean).query(
        "split == 'random' and policy == 'validation'")[
            ["model", "threshold", "balanced_accuracy", "roc_auc",
             "average_precision", "f1", "brier", "ece10", "action_rate"]]
    t2 = summary.query("rate == 0.30 and model != 'Constant baseline'")[
        ["model", "mechanism", "dfr_mean", "dfr_ci_low", "dfr_ci_high",
         "dss_mean", "harmful_mean", "beneficial_mean",
         "delta_balanced_accuracy_mean", "delta_roc_auc_mean"]]
    t1.to_csv(out / "table1.csv", index=False)
    t2.to_csv(out / "table2.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    for j, kind in enumerate(("missing", "noise")):
        for name in NAMES:
            g = summary.query("model == @name and mechanism == @kind"
                              ).sort_values("rate")
            x = g.rate.to_numpy() * 100
            axes[0, j].plot(x, g.dfr_mean.to_numpy() * 100, marker="o",
                            color=COLORS[name], label=name,
                            linestyle="--" if name == NAMES[3] else "-")
            axes[0, j].fill_between(
                x, g.dfr_ci_low.to_numpy() * 100,
                g.dfr_ci_high.to_numpy() * 100, color=COLORS[name], alpha=0.12)
            axes[1, j].plot(x, g.balanced_accuracy_mean.to_numpy(), marker="o",
                            color=COLORS[name],
                            linestyle="--" if name == NAMES[3] else "-")
        axes[0, j].set_title(("a  " if j == 0 else "b  ") + kind.capitalize())
        axes[1, j].set_title("c  Missingness" if j == 0 else "d  Noise")
        axes[1, j].set_xlabel("Nominal selected-cell rate (%)")
        for ax in axes[:, j]:
            ax.set_xticks([5, 10, 20, 30])
            ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Decision flip rate (%)")
    axes[1, 0].set_ylabel("Balanced accuracy")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figure1.pdf", bbox_inches="tight")
    fig.savefig(out / "figure1.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    g = t2.reset_index(drop=True)
    x = np.arange(len(g))
    labels = [KEY[r.model].upper() + "\n" +
              ("Missing" if r.mechanism == "missing" else "Noise")
              for r in g.itertuples()]
    axes[0].bar(x, 100 * g.harmful_mean, color="#D55E00", label="Harmful")
    axes[0].bar(x, 100 * g.beneficial_mean, bottom=100 * g.harmful_mean,
                color="#009E73", label="Beneficial")
    axes[0].set_xticks(x, labels, fontsize=7)
    axes[0].set_ylabel("Fraction of all cases (%)")
    axes[0].set_title("a  Decision flips at 30% corruption")
    axes[0].legend(fontsize=7)
    a = assoc.set_index(["model", "mechanism"]).loc[
        list(zip(g.model, g.mechanism))].reset_index()
    axes[1].bar(x - 0.18, a.flip_error_auc, 0.36,
                color="#0072B2", label="Flip frequency")
    axes[1].bar(x + 0.18, a.margin_error_auc, 0.36,
                color="#E69F00", label="Negative clean margin")
    axes[1].axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    axes[1].set_xticks(x, labels, fontsize=7)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("ROC-AUC for clean-error discrimination")
    axes[1].set_title("b  Exploratory error association")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figure2.pdf", bbox_inches="tight")
    fig.savefig(out / "figure2.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return t1, t2, assoc, summary


def md_table(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(
            f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
            for v in row) + " |")
    return "\n".join(lines)


def write_result_sections(t1, t2, assoc, summary, meta, args, out):
    best = t1.sort_values("balanced_accuracy", ascending=False).iloc[0]
    learned = t2.dfr_mean.to_numpy()
    abstract = (
        "Aggregate predictive performance does not identify which individual "
        "decisions change when input quality deteriorates. We evaluate fixed "
        "binary classifiers on 41,188 Bank Marketing records after removing "
        "call duration. Training, validation and test partitions isolate model "
        "fitting, threshold selection and evaluation. Logistic Regression, "
        "Random Forest and HistGradientBoosting are compared with a constant "
        "baseline under 5%, 10%, 20% and 30% missingness or feature noise, "
        f"with {args.reps} paired repetitions. Decision flip rate is decomposed "
        "into harmful and beneficial changes and assessed alongside predictive "
        "and calibration metrics. At 30% corruption in the random-split "
        f"benchmark, learned-model flip rates range from {100 * learned.min():.2f}% "
        f"to {100 * learned.max():.2f}%. {best['model']} has the highest observed "
        f"clean balanced accuracy ({best['balanced_accuracy']:.3f}) among the "
        "evaluated models. Results apply to this dataset, fitted models and "
        "specified perturbations; case-bootstrap intervals condition on the "
        "realized corruption draws. The audit reports action changes separately "
        "from net predictive changes and does not equate stability with correctness."
    )
    result = (
        "Table 1 reports clean-test performance using thresholds selected only "
        "on validation data. Table 2 reports 30% corruption, the pre-specified "
        "most severe stress condition, rather than a condition selected for "
        "statistical significance. Figure 1 presents all four severity levels.\n\n"
        f"At 30%, observed learned-model DFR ranges from "
        f"{100 * learned.min():.2f}% to {100 * learned.max():.2f}%. "
        "For each row, DFR equals harmful plus beneficial flips, and DSS equals "
        "one minus DFR. The uncertainty intervals resample cases after averaging "
        "over corruption repetitions; they exclude model-training uncertainty "
        "and do not constitute unconditional population confidence intervals.\n\n"
        "Figure 2 separates harmful and beneficial changes and compares "
        "flip-frequency discrimination of clean errors with negative clean "
        "threshold margin. This analysis is exploratory and uses stress-test "
        "information unavailable without clean reference predictions. It is not "
        "an independently validated deployment-time error detector. Threshold "
        "0.5 and chronological-split results are retained in all_metrics.csv; "
        "they must be interpreted alongside the primary random-split results."
    )
    template = Path(__file__).with_name("manuscript.md").read_text()
    text = (template.replace("{{ABSTRACT}}", abstract)
            .replace("{{RESULTS}}", result)
            .replace("{{TABLE1}}", md_table(t1))
            .replace("{{TABLE2}}", md_table(t2))
            .replace("{{REPETITIONS}}", str(args.reps)))
    status_start = text.index("**Document status:**")
    status_end = text.index("\n\n## Abstract", status_start)
    text = (text[:status_start] +
            "**Document status:** Populated from a completed experiment. "
            "Independent reproduction, scientific interpretation, reference "
            "verification and journal-format review remain required." +
            text[status_end:])
    if "{{" in text:
        raise ValueError("Unresolved manuscript placeholders.")
    (out / "manuscript_with_computed_results.md").write_text(text)
    meta["abstract_word_count"] = len(abstract.split())
    assert meta["abstract_word_count"] <= 250


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Local bank-additional-full.csv; else download UCI")
    parser.add_argument("--out", default="results")
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--random-only", action="store_true",
                        help="Diagnostic only; omits chronological sensitivity")
    args = parser.parse_args()
    if args.reps < 3 or args.bootstrap < 100:
        parser.error("At least 3 corruption repetitions and 100 bootstraps required.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use an empty output folder to prevent stale artifacts.")
    started = time.time()
    data, y, meta = read_data(args.data, out)
    records, clean, cases, arrays = [], [], {}, {}
    meta.update({"title": TITLE, "seed": args.seed, "repetitions": args.reps,
                 "bootstrap_draws": args.bootstrap,
                 "noise_sd_multiplier": 0.25, "severity_levels": LEVELS,
                 "python": sys.version, "platform": platform.platform(),
                 "numpy": np.__version__, "pandas": pd.__version__,
                 "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                 "matplotlib": matplotlib.__version__,
                 "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "splits": {}, "constant_threshold": 0.5})
    for split_id, split in enumerate(
            ("random",) if args.random_only else ("random", "chronological")):
        print("Split:", split, flush=True)
        tr, va, te = partitions(y, args.seed, split)
        meta["splits"][split] = {
            "train_n": len(tr), "validation_n": len(va), "test_n": len(te),
            "train_prevalence": float(y[tr].mean()),
            "validation_prevalence": float(y[va].mean()),
            "test_prevalence": float(y[te].mean())}
        for label, ids in (("train", tr), ("validation", va), ("test", te)):
            arrays[f"{split}_{label}_row_ids"] = ids
        train, validation, test = data.iloc[tr], data.iloc[va], data.iloc[te]
        transformer, numeric, categorical = preprocessor(train)
        xt = transformer.fit_transform(train)
        xv = transformer.transform(validation)
        x0 = transformer.transform(test)
        assert "duration" not in train
        spec0 = corruption_spec(train, numeric, categorical)
        for zero_kind in ("missing", "noise"):
            zero, zero_info = corrupt(test, numeric, categorical, spec0, 0,
                                      zero_kind, args.seed)
            np.testing.assert_allclose(transformer.transform(zero), x0,
                                       rtol=0, atol=0)
            assert zero_info["realized_cell_fraction"] == 0
        models = fit_models(xt, y[tr], args.seed, args.jobs)
        reference, thresholds = {}, {}
        for name, model in models.items():
            reference[name] = model.predict_proba(x0)[:, 1]
            thresholds[name] = (0.5 if name == NAMES[3] else
                                choose_threshold(y[va], model.predict_proba(xv)[:, 1]))
            arrays[f"{split}_{KEY[name]}_clean_probability"] = reference[name]
            for policy, threshold in (("validation", thresholds[name]), ("fixed_05", 0.5)):
                values, _ = metrics(y[te], reference[name], threshold, reference[name])
                assert values["dfr"] == 0
                clean.append({"split": split, "model": name, "policy": policy,
                              "threshold": threshold, **values})
        spec = corruption_spec(train, numeric, categorical)
        meta["splits"][split]["corruption_spec"] = spec
        for kind_id, kind in enumerate(("missing", "noise")):
            for rate in LEVELS:
                totals = {name: np.zeros(len(te), dtype="float32") for name in NAMES}
                for repetition in range(args.reps):
                    seed = args.seed + split_id * 100000 + kind_id * 10000 + repetition
                    bad, corruption_info = corrupt(
                        test, numeric, categorical, spec, rate, kind, seed)
                    xb = transformer.transform(bad)
                    for name, model in models.items():
                        p = model.predict_proba(xb)[:, 1]
                        for policy, threshold in (
                                ("validation", thresholds[name]), ("fixed_05", 0.5)):
                            values, flips = metrics(y[te], p, threshold, reference[name])
                            records.append({
                                "split": split, "model": name, "mechanism": kind,
                                "rate": rate, "repetition": repetition,
                                "corruption_seed": seed, "policy": policy,
                                "threshold": threshold, **corruption_info, **values})
                            if policy == "validation":
                                totals[name] += flips
                for name in NAMES:
                    prefix = f"{split}_{KEY[name]}_{kind}_{int(rate * 100)}"
                    frequency = totals[name] / args.reps
                    clean_error = (reference[name] >= thresholds[name]) != y[te]
                    margin = np.abs(reference[name] - thresholds[name])
                    cases[(split, name, kind, rate)] = {
                        "frequency": frequency, "clean_error": clean_error,
                        "margin": margin}
                    arrays[prefix + "_flip_frequency"] = frequency
                    arrays[f"{split}_{KEY[name]}_clean_error"] = clean_error
                    arrays[f"{split}_{KEY[name]}_margin"] = margin
                print(f"  completed {kind} at {rate:.0%}", flush=True)
    arrays["target"] = y
    np.savez_compressed(out / "case_audit.npz", **arrays)
    t1, t2, association, summary = tables_and_figures(
        records, clean, cases, args, out)
    write_result_sections(t1, t2, association, summary, meta, args, out)
    meta["elapsed_seconds"] = time.time() - started
    meta["primary_metric_rows"] = len(records)
    meta["diagnostic_random_only"] = args.random_only
    meta["execution_complete"] = True
    meta["submission_ready"] = False
    meta["remaining_review"] = [
        "Independent reproduction and manuscript scientific review",
        "Interpret chronological sensitivity and fixed-threshold results",
        "Official template and accepted-paper comparison",
        "DOCX/PDF formatting, page-count verification and visual QA",
        "Reference full-text verification and complete novelty review",
        "Q1/Scopus ranking verification for the named year and category",
        "Author declarations, affiliation and repository publication"]
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print("Experiment completed. Results:", out.resolve())
    print("Generated manuscript requires editorial and scientific review; "
          "it is not certified submission-ready.")


if __name__ == "__main__":
    main()
