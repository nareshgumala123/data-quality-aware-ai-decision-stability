# Supplementary Material S1: Reproducibility and analytical details

Naresh Gumala. Data Quality-Aware AI Decision Intelligence.

## Execution

Use Python 3.11 or 3.12 in a fresh environment. From the repository root:

```bash
python -m pip install -r requirements.txt
python source/research.py --reps 30 --bootstrap 500 --out rerun
python audit_results.py
```

The first command pins five scientific packages. The experimental script downloads the public UCI ZIP, selects bank-additional-full.csv and removes duration. Alternatively, pass `--data /path/bank-additional-full.csv`. The supplied original results are preserved under results/; the full run above writes to rerun/. The audit reads results/ and writes the two final figure files and expanded table CSVs under final/. It validates the original experimental source hash. Run the audit under the pinned environment to match publication rendering closely. No private data, API credentials or paid service is required.

For this repository, use the command-line instructions in README.md. The historical Colab launcher is not included. After a full run, compare the new outputs with the supplied results using `python scripts/verify_reproduction.py --reference results --candidate rerun`. The audit command above checks the supplied results; the comparison command checks the fresh run.

## Fixed design

The original research.py is unchanged: source SHA-256 d9aa3ebbe7519e2c3d958a3c49c737c29eed40ee5f706dee16bba2ac50440ebe. Input CSV SHA-256: 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8. UCI Bank Marketing DOI: https://doi.org/10.24432/C5K306 ; CC BY 4.0, credited to Moro, Rita and Cortez (2014).

Both partitions contain 24,712 training, 8,238 validation and 8,238 test cases. Random stratification uses seed 20261023; the chronological sensitivity respects supplied file order. Recorded row identifiers are in case_audit.npz. One training fit is performed per model and split. Numerical median imputation and standardization and categorical mode imputation and dense one-hot encoding are training-only. The observed unknown category is retained; unseen categorical levels are ignored by the encoder. No test labels select thresholds or preprocessing.

Logistic Regression: C=1, lbfgs, max_iter=3000. Random Forest: 200 trees, min_samples_leaf=5, max_features=sqrt, two jobs. HistGradientBoosting: 150 iterations, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1, early_stopping=False. Other settings are library defaults under the pinned version. All learned models share the same transformed inputs. The prior-probability dummy at threshold 0.5 is the constant-decision control. Learned thresholds maximize validation balanced accuracy over ROC thresholds; ties within 1e-12 choose proximity to 0.5. A separate fixed-0.5 policy is saved for sensitivity analysis.

## Corruption and measurement

Missing and noise conditions are separate. Each selects cells independently at 5%, 10%, 20% or 30%, with 30 paired seeds. Severity masks are nested within seed; each model evaluates the same corrupted records. Numeric noise is Gaussian with SD 0.25 times training-column SD. Count-like fields are rounded and constrained as coded. The pdays=999 sentinel is protected and excluded from its scale estimate. Categorical noise selects a different member of the training vocabulary. Noise selection is not identical to realized value change. At 30% nominal selection, realized changes average 25.77% for random and 25.89% for chronological test data. Missingness is controlled MCAR; MAR and MNAR robustness is not tested.

DFR is the proportion of reference binary decisions that change. DSS=1−DFR. Harmful flips turn clean-correct cases into errors; beneficial flips repair clean errors. Both use all test cases as denominator. Thus DFR=H+B and delta ordinary accuracy=B−H. Balanced accuracy uses class-specific weights and is not interchangeable with ordinary accuracy. ROC-AUC and average precision use probabilities, not the chosen decision threshold. ECE uses ten fixed equal-width probability bins; Brier score is the mean squared probability error. Metrics are defined directly in research.py.

## Uncertainty and exploratory analyses

Primary random-split 95% DFR intervals use 500 case bootstraps after averaging each case’s flip indicator over corruptions. They condition on the fitted model, chosen threshold and realized perturbations; they exclude training uncertainty and do not imply temporal independence of all customer observations. Chronological curves and table values are descriptive without these intervals. Repetitions summarize perturbation variability, not 30 independent model-training experiments.

The severity Spearman coefficient is computed across four condition means. All six learned-model random-split pairs yield 1.0; this limited monotonic check is not a confirmatory significance test. Clean-error association uses case flip frequency as an exploratory score and compares it with clean errors. ROC-AUC ranges from 0.3716 (RF noise) to 0.7900 (HGB missingness). An inverse association in RF noise rules out a universal error flag. The negative clean-margin comparator and Mann–Whitney/Holm outputs are in error_association.csv. Machine underflow can store extremely small p-values as zero; these are not reported as p=0 and are not the manuscript’s evidence for novelty.

At fixed threshold 0.5, random-split 30% missingness/noise DFRs are 6.29%/2.43% (LR), 2.45%/1.85% (RF) and 2.59%/2.19% (HGB). Their corrupted balanced accuracies are 0.596/0.590, 0.557/0.585 and 0.561/0.593, respectively. Lower turnover at another threshold does not establish improved utility; policy choice changes the reference actions.

Fixed-threshold and chronological sensitivities, including action rate, calibration, F1, probability shifts, achieved cell rates and all repetition-level observations, are retained in all_metrics.csv. Clean policy metrics are in clean_metrics.csv. These should be assessed before selecting any operational policy. A threshold of 0.5 does not imply cost optimality for the imbalanced target.

## Files and provenance

results/ contains the unchanged nine essential numeric and metadata files selected from the original 14-file result archive; historical rendered figures and the automatically populated manuscript are omitted. source/research.py is the unchanged implementation that generated it. source/manuscript.md remains the historical template required by that script; it is not the final journal manuscript. The edited Word manuscript and its text companion remain in the original research package and are not part of this executable repository. Original summary tables in results/ are provenance outputs; the two publication figures and expanded two tables in final/ are regenerated by audit_results.py from the same observations.

The original numerical and reproduction audit reports in final/supplement/ are preserved historical records. The supplied reproduction report records an earlier full run with exact decision counts, maximum scalar difference 3.4547278704444295e-8 and maximum probability difference 5.020650561959883e-12. Those numbers describe that earlier report, not a new execution. Current repository verification is recorded separately in verification/. The original experiment and its run metadata remain unchanged. The public repository provides executable code and essential results; this does not certify manuscript submission readiness or broad scientific validity.

## Interpretation limits

This is a historical bank-marketing benchmark, not enterprise readiness or a knowledge-rule inference model. Duration exclusion does not verify every retained feature’s availability in a real contact workflow. Corruptions simulate measurement defects while labels remain fixed. One dataset, one training seed, fixed hyperparameters, shared imputation and limited business-utility measurement restrict generalization. Flips are neither automatically harmful nor proof of unfairness. Additional datasets and a clearer methodological gap are required before claiming broad novelty or deployment readiness.
