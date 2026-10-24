# Data Quality-Aware AI Decision Intelligence

**Paper:** Data Quality-Aware AI Decision Intelligence: Quantifying Decision Stability Under Missing and Noisy Data  
**Author:** Naresh Gumala  
**Status:** Public research code and results; this repository does not certify journal acceptance or submission readiness.

Paired evaluation of decision turnover under missing and noisy input features on UCI Bank Marketing (41,188 records). The experiment compares Logistic Regression, Random Forest, HistGradientBoosting and a constant baseline at 5%, 10%, 20% and 30% corruption, with 30 paired repetitions and both random and chronological partitions. It reports decision flip rate (DFR), harmful and beneficial flips, predictive performance and calibration.

## Upload status

**The three requested result uploads are complete:** [all_metrics.csv](results/all_metrics.csv), [case_audit.npz](results/case_audit.npz) and [figure1.png](results/figure1.png). These files are preserved byte-for-byte from `GitHub_Remaining_3_Files.zip`. The existing code and results are unchanged. The existing [Figure 1 PDF](final/figures/Figure_1.pdf) is also present. No Figure 2 was added in this completion; the pre-existing historical Figure 2 PDF remains unchanged and is excluded from the final manuscript. Upload completion does not mean strict numerical reproduction passed. See [upload record](verification/upload_status.json).

## Verification status

The saved-result audit passed on 23 October 2026. A fresh full run with the five pinned package versions completed both partitions and all 30 repetitions, but **strict reproduction failed for Random Forest** in both partitions. Logistic Regression, HistGradientBoosting and the constant baseline retained identical saved decision frequencies and clean error flags; Random Forest changed 26 random-partition and 225 chronological-partition clean error flags. Maximum Random Forest clean-probability differences were 0.03178 and 0.02348, respectively. The cause has not been established; platform/build differences remain a hypothesis, not a verified explanation. This upload did not rerun training or resolve the discrepancy. See [current comparison](verification/reproducibility.json), [model breakdown](verification/model_comparison.json) and [fresh run metadata](verification/fresh_run_metadata.json). The earlier package report of exact reproduction is historical evidence and is not confirmed by this new run.

## Reproduce

Use **Python 3.11 or 3.12**. Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python audit_results.py
python source/research.py --reps 30 --bootstrap 500 --out rerun
python scripts/verify_reproduction.py --reference results --candidate rerun
```

On Windows activate the environment with `.venv\Scripts\activate`. The audit validates the supplied `results/`, including the original source hash, 3,840 evaluation rows, 16 clean-policy rows, partition separation, paired corruption and flip identities. It regenerates two figures and two expanded tables under `final/`; it does not train models. For its checks, run Python normally without `-O`.

The full experiment downloads the public UCI archive and trains both partitions. For an existing dataset use `--data /path/to/bank-additional-full.csv`. `--out rerun` keeps the supplied results intact; the output directory must be empty. The comparison command checks seven CSVs and all 95 case arrays, requiring identical saved decision frequencies, row IDs, labels and error flags, with absolute tolerance `5e-8` for other numeric values. It exits with a failure status when comparisons fail. Figure bytes and environment metadata can differ across systems.

## Files

| Path | Purpose |
| --- | --- |
| [source/research.py](source/research.py) | Original experiment, unchanged |
| [source/manuscript.md](source/manuscript.md) | Historical text template required by the experiment; not a submission manuscript |
| [audit_results.py](audit_results.py) | Original saved-result audit, unchanged |
| [requirements.txt](requirements.txt) | Five original pinned scientific dependencies; mirrored in `source/requirements.txt` |
| [results/](results/) | Original metric CSVs, compressed case arrays and run metadata |
| [results/figure1.png](results/figure1.png) | Figure 1 PNG from the prepared remaining-upload package |
| [final/figures/Figure_1.pdf](final/figures/Figure_1.pdf) | Existing publication Figure 1 PDF |
| [final/](final/) | Existing publication layouts, expanded tables and historical audit reports; Figure 2 is excluded from the final manuscript |
| [Supplementary_Material_S1.md](Supplementary_Material_S1.md) | Methods, interpretation limits and reproduction details |
| [verification/](verification/) | Current verification evidence and provenance |
| [scripts/verify_reproduction.py](scripts/verify_reproduction.py) | Comparison helper added for this repository |

The original source SHA-256 is `d9aa3ebbe7519e2c3d958a3c49c737c29eed40ee5f706dee16bba2ac50440ebe`. Its opening comment predates the completed runs. The original run metadata and earlier audit reports are retained as historical records; current verification is recorded separately under `verification/`. The publication artwork is preserved as supplied; rerendering with the audit may change image/PDF bytes. This repository includes the executable subset of the research package, without the Word manuscripts or historical auto-generated figures.

## Data and interpretation

Dataset: [Moro, Rita and Cortez, UCI Bank Marketing](https://doi.org/10.24432/C5K306), CC BY 4.0. The input is `bank-additional-full.csv`; the call-duration predictor is removed. Expected input CSV SHA-256: `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`. The raw dataset is downloaded from UCI rather than redistributed here; case arrays retain row indexes, target labels and model outputs needed for auditing.

The study uses one dataset, one training seed and synthetic perturbations. Case-bootstrap intervals condition on the fitted model and realized corruptions; they do not quantify training uncertainty. Decision changes can be harmful or beneficial. The results do not establish business utility, broad novelty, or deployment readiness.

## Attribution

Naresh Gumala. ChatGPT assisted with implementation, analysis review and writing. The author has not selected a software license; the dataset's CC BY 4.0 license does not assign a license to this code.
