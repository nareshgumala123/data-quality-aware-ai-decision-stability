# Data Quality-Aware AI Decision Intelligence: Quantifying Decision Stability Under Missing and Noisy Data

Naresh Gumala

DBA 802 Data Analytics and Strategic Decision Intelligence

**Document status:** Research manuscript source. Experiments have not been executed in this session. Bracketed generation fields are deliberately unfilled. This file is not a completed empirical paper or a submission-ready manuscript. The experiment fills these fields from its own outputs.

## Abstract

{{ABSTRACT}}

## Keywords

Artificial intelligence; decision intelligence; data quality; decision stability; missing data; noisy data

## 1 Introduction

AI-supported decisions connect predictive models to organizational action. In a marketing workflow, the same aggregate success rate can accompany a change in which customers receive attention. Model evaluation therefore needs to distinguish average predictive performance from the consistency of case-level actions. Data-quality research motivates evaluating defects across the data lifecycle [1-4], while missing-data theory distinguishes mechanisms that cannot be treated as interchangeable [5].

Prediction disagreement is an established research topic. Prior work examines churn across retrained models [7]. The present study fixes each trained model and its decision threshold, then perturbs the inputs presented at evaluation. Decision flip rate and its complement are used as transparent audit summaries, not introduced as novel mathematical metrics. The intended contribution is a reproducible evaluation of action instability, its harmful and beneficial components, and its relationship to aggregate predictive changes in a managerial benchmark.

Two questions guide the analysis. RQ1 asks how increasing missingness and feature noise affect fixed-model binary decisions. RQ2 asks whether case-level action changes provide information that is not determined by balanced accuracy and ROC-AUC. RQ2 concerns complementary information, not a presumption that stability is a superior objective.

## 2 Methods

### 2.1 Data and decision setting

The study uses bank-additional-full.csv, the 41,188-record version of the UCI Bank Marketing dataset, associated with term-deposit subscription [6,9]. The target is subscription rather than customer benefit or campaign profitability. Call duration is excluded because it is unavailable before the call concludes. The remaining 19 predictors form a duration-excluded targeting benchmark. Exclusion of duration alone does not establish that every remaining campaign variable is available at the start of a campaign.

Observed categorical values labelled unknown are retained as categories. They are distinct from experimentally introduced missing cells. Labels are unchanged under corruption: the experiment models defective measurement of a fixed case, not an intervention that changes the customer's outcome. UCI describes the additional-full file as ordered by date; this ordering permits a chronological sensitivity analysis [9].

### 2.2 Fitting and threshold selection

The primary split allocates approximately 60%, 20% and 20% of records to stratified training, validation and test sets. A second analysis assigns these fractions in file order without shuffling. Median imputation and standardization of numerical variables, and mode imputation and dense one-hot encoding of categorical variables, are fitted exclusively on training data. Unseen categories are ignored by the encoder.

Logistic Regression, Random Forest and HistGradientBoosting use fixed hyperparameters documented in research.py. No model is selected or retuned on corrupted test data. HistGradientBoosting uses the common imputed feature representation rather than its native missing-value mechanism; results therefore compare complete pipelines, not the maximum attainable missingness robustness of each algorithm.

For each learned model, the threshold maximizes validation balanced accuracy. Ties are resolved by proximity to 0.5, with deterministic array order breaking any remaining tie. Thresholds remain fixed throughout test corruption. A fixed-0.5 sensitivity analysis is also saved. A prior-probability constant classifier uses threshold 0.5 as a negative control. A constant decision can be perfectly stable while providing little discrimination.

### 2.3 Paired corruption design

Missingness and noise are evaluated separately at nominal selected-cell rates of 5%, 10%, 20% and 30%, with {{REPETITIONS}} repetitions. Missingness masks are sampled independently of values and outcomes, representing a controlled missing-completely-at-random mechanism [5]. These results do not establish robustness to missing-at-random or missing-not-at-random mechanisms.

For noise, selected numerical cells receive independent Gaussian perturbations with standard deviation equal to 0.25 times the training-column standard deviation. Count variables are rounded and constrained to non-negative values; age is constrained to positive values. The pdays value 999 is protected as a sentinel and excluded from estimation of that column's noise scale. Categorical noise replaces a selected value with a different category drawn uniformly from the training vocabulary. Nominal rate controls cell selection, not Gaussian amplitude. Attempted, realized and eligible cell fractions are recorded because rounding or protected cells can change the achieved noise dose.

Masks are nested across severity levels within each repetition. All models evaluate identical corrupted cases. Preprocessing parameters, labels and decision thresholds are held fixed. Corruption creates a controlled stress test; independent cell errors may disrupt real feature relationships and are not claimed to reproduce a bank's operational defect distribution.

### 2.4 Outcomes and analysis

Let p_i be the reference probability, p_i' its corrupted counterpart, t the frozen threshold and y_i the observed outcome. Define d_i = 1[p_i >= t] and d_i' = 1[p_i' >= t]. The decision flip rate is

DFR = n^(-1) sum_i 1[d_i != d_i'],

and DSS = 1 - DFR. DSS is a complementary scale, not an independent outcome. Harmful and beneficial flip rates use all test cases as their denominator:

H = n^(-1) sum_i 1[d_i = y_i and d_i' != y_i],

B = n^(-1) sum_i 1[d_i != y_i and d_i' = y_i].

For binary labels, DFR = H + B and corrupted accuracy minus reference accuracy equals B - H. Thus net accuracy change can conceal offsetting action changes. Balanced accuracy weights classes differently, so its change is reported separately; the accuracy identity must not be substituted for a balanced-accuracy identity.

The audit also reports balanced accuracy, ROC-AUC, average precision, F1, Brier score, action rate, signed and absolute probability shifts, and positive-class calibration error in ten fixed equal-width bins. Calibration error depends on binning and is interpreted alongside Brier score [8].

Corruption repetitions characterize perturbation variability; they are not independent replications of model training or independent test cohorts. Primary DFR intervals resample cases after averaging flip indicators over repetitions. These conditional intervals do not capture training-sample uncertainty or the full uncertainty over corruption generation. Chronological results are treated descriptively because ordered cases can be dependent.

Spearman coefficients summarize the four severity-level mean DFRs without inferential p-values. At 30% corruption, exploratory analyses compare per-case flip frequency between clean-error and clean-correct groups using Mann-Whitney tests, with Holm adjustment across the six model-mechanism tests. Error-discrimination ROC-AUC is compared with negative absolute clean threshold margin. These analyses do not validate a deployable error detector because they require clean-reference and stress-test information.

## 3 Results

{{RESULTS}}

### Table 1 Clean-test predictive performance

{{TABLE1}}

All values are computed on the untouched random-split test partition. Thresholds for learned models come from validation balanced accuracy; the constant baseline uses 0.5. Average precision is included because the outcome is imbalanced. Higher is preferable for balanced accuracy, ROC-AUC, average precision and F1; lower is preferable for Brier score and calibration error. Action rate is descriptive.

### Figure 1 Decision stability and predictive performance across corruption severity

![Figure 1](figure1.png)

Random-split decision flip rate and balanced accuracy under missingness and noise. Lines show means over repeated corruptions. DFR bands are 95% case-bootstrap intervals conditional on the fitted model, fixed partition and realized corruption draws. Rates are nominal selected-cell rates; realized noise fractions are retained in the supplementary outputs. The constant baseline illustrates why stability alone is insufficient.

### Table 2 Decision changes at 30 percent nominal corruption

{{TABLE2}}

DFR, DSS, harmful and beneficial rates use the full test-set denominator. DFR intervals are conditional case-bootstrap intervals. Delta balanced accuracy and delta ROC-AUC are corrupted minus reference values; negative values indicate deterioration. DFR equals harmful plus beneficial flips. The table reports the pre-specified most severe condition, not an outcome-selected condition.

### Figure 2 Direction of flips and exploratory association with clean errors

![Figure 2](figure2.png)

Panel a decomposes flips into clean-correct to corrupted-incorrect changes and clean-incorrect to corrupted-correct changes. Panel b compares flip-frequency ROC-AUC for identifying clean errors with the negative-margin comparator. These are in-sample exploratory associations at the pre-specified 30% stress condition, not prospective error-detection performance. Association values are not used to tune classifiers or select thresholds.

## 4 Discussion

A stability audit concerns the identity of actions as well as average outcomes. The decomposition clarifies why unchanged accuracy cannot establish unchanged decisions. Conversely, low flip rates cannot establish correctness: a constant classifier supplies a direct counterexample. Managerial interpretation therefore requires joint examination of predictive performance, action rates, probability quality and directional changes.

For a bank, harmful flips may represent missed subscriptions or additional unsuccessful contacts, depending on their direction and the action policy. The dataset contains no intervention costs or incremental treatment effects, so no profit, return-on-investment or causal campaign-benefit claim follows. A balanced-accuracy threshold is a diagnostic convention; a deployment policy would need explicit costs, capacity and customer-impact criteria.

The comparison addresses one historical dataset, fixed model configurations and a single training seed per split. Chronological sensitivity tests dependence on a random-split design but does not constitute multi-institutional validation. Records may contain repeated contacts, and the available data do not support a customer-group split. Imputation can erase missingness information, the noise mechanism is deliberately simplified, and numerical and categorical noise rates are not equivalent physical error magnitudes.

A stronger journal contribution would require a focused comparison with existing input-robustness and prediction-consistency research, additional independent datasets, and evidence that the audit changes a useful operational decision. Merely renaming disagreement measures does not supply that contribution. Further work could evaluate naturally occurring defects, conditional corruption mechanisms, model-refitting variability and cost-sensitive review policies.

## 5 Conclusion

Decision stability is a useful companion to predictive evaluation when input quality changes, provided that stability is not interpreted as correctness. A reproducible audit should freeze the decision policy, pair corrupted cases across models, report the achieved corruption dose, and distinguish harmful from beneficial flips. Empirical conclusions must remain bounded by the executed experiment and its sensitivity analyses.

## Declarations to complete before submission

The author must confirm institutional affiliation, corresponding-author details, funding, competing interests, author contributions and the applicable ethics requirements. Public-data licensing does not by itself determine whether an institution requires an ethics determination. AI assistance in study design, programming and manuscript preparation must be disclosed as required by the journal and course. The author must review and take responsibility for the work.

Data availability: UCI Bank Marketing, https://doi.org/10.24432/C5K306, CC BY 4.0. Code availability: repository publication is pending. No repository URL is asserted.

## References

[1] M. Hagn, B. Heinrich, T. Krapf, A. Schiller, Handling imperfection: A taxonomy for machine learning on data with data quality defects, Decision Support Systems 196 (2025) 114493. https://doi.org/10.1016/j.dss.2025.114493.

[2] Y. Gong, G. Liu, Y. Xue, R. Li, L. Meng, A survey on dataset quality in machine learning, Information and Software Technology 162 (2023) 107268. https://doi.org/10.1016/j.infsof.2023.107268.

[3] S.E. Whang, Y. Roh, H. Song, J.-G. Lee, Data collection and quality challenges in deep learning: a data-centric AI perspective, The VLDB Journal 32 (2023) 791-813. https://doi.org/10.1007/s00778-022-00775-9.

[4] P. Singh, Systematic review of data-centric approaches in artificial intelligence and machine learning, Data Science and Management 6 (2023) 144-157. https://doi.org/10.1016/j.dsm.2023.06.001.

[5] R.J.A. Little, D.B. Rubin, Statistical Analysis with Missing Data, third ed., Wiley, 2019. https://doi.org/10.1002/9781119482260.

[6] S. Moro, P. Cortez, P. Rita, A data-driven approach to predict the success of bank telemarketing, Decision Support Systems 62 (2014) 22-31. https://doi.org/10.1016/j.dss.2014.03.001.

[7] D. Bahri, H. Jiang, Locally Adaptive Label Smoothing Improves Predictive Churn, Proceedings of the 38th International Conference on Machine Learning, PMLR 139 (2021) 532-542. https://proceedings.mlr.press/v139/bahri21a.html.

[8] C. Guo, G. Pleiss, Y. Sun, K.Q. Weinberger, On Calibration of Modern Neural Networks, Proceedings of the 34th International Conference on Machine Learning, PMLR 70 (2017) 1321-1330. https://proceedings.mlr.press/v70/guo17a.html.

[9] S. Moro, P. Rita, P. Cortez, Bank Marketing [Dataset], UCI Machine Learning Repository, 2014. https://doi.org/10.24432/C5K306.

