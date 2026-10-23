# Data Quality-Aware AI Decision Intelligence

**Paper:** Data Quality-Aware AI Decision Intelligence: Quantifying Decision Stability Under Missing and Noisy Data  
**Author:** Naresh Gumala  
**Status:** Research code repository under preparation; paper is not published.

## Study
Paired evaluation of AI decision turnover under missing and noisy input features on the UCI Bank Marketing dataset (41,188 records). Models: Logistic Regression, Random Forest, Histogram-Based Gradient Boosting, and a constant baseline. Corruption levels: 5%, 10%, 20%, 30%; 30 repetitions; random and chronological partitions. Outputs include decision flip rate (DFR), harmful and beneficial flips, balanced accuracy, and ROC AUC.

## Data
UCI Bank Marketing: https://doi.org/10.24432/C5K306 . The experiment uses `bank-additional-full.csv` and removes the call-duration predictor. Download from https://archive.ics.uci.edu/dataset/222/bank+marketing . Original data are not redistributed here.

## Reproducibility
**The source code and result artifacts are not yet uploaded to this repository.** Do not claim this repository reproduces the paper until those files have been committed and checked. Intended experiment configuration: seed 20261023, 30 perturbation repetitions, 500 bootstrap draws, one training seed per partition. Conditional uncertainty intervals are not across independent retraining runs.

## Limitations
One dataset, synthetic missingness/noise, one training seed, no measured decision utility or real-world deployment validation.

## Attribution
Naresh Gumala. ChatGPT assisted with implementation, analysis review, and manuscript writing. No software license is granted until selected by the author.
