# Privacy vs Fairness in Machine Learning

This repository studies the tension between differential privacy and group fairness in classification tasks. We investigate how adding Laplace noise to training data affects fairness metrics (Equal Opportunity, Equalized Odds) and whether reweighting techniques can mitigate bias while preserving privacy.

## Overview

Enforcing differential privacy during model training introduces noise that can exacerbate existing disparities across sensitive attributes. This work examines whether in-processing bias mitigation methods remain effective under privacy constraints, and quantifies the utility-fairness trade-offs.

**Key Questions:**
- How does the privacy budget (ε) affect Equal Opportunity and Equalized Odds?
- Can reweighting maintain fairness after DP noise injection?
- What is the resulting accuracy trade-off?

## Repository Structure

```
/src
  /data_loading        Dataset loaders and preprocessing (Adult, COMPAS, CelebA, FolkTables)
  /models              Model definitions and training loops
  /metrics             Fairness metrics (EOD, AOD) and privacy parameter tracking
  /experiments         Experiment runners and CLI entrypoints
  /utils               Config loading, logging, seeding
/notebooks             Exploratory analysis (archive)
/results               Timestamped experimental outputs (CSV, pickles, plots)
/data                  Datasets (excluded from git if large)
```

## Getting Started

**Requirements:**
- Python 3.8+
- See `requirements.txt` for dependencies

**Install:**
```bash
pip install -r requirements.txt
```

**Run a baseline experiment:**
```bash
python -m src.experiments.train --dataset adult --epsilon 1.0 --mitigation reweighing
```

**Run privacy-fairness sweep:**
```bash
python -m src.experiments.privacy_curve --dataset folktables --epsilon_min 0.001 --epsilon_max 10.0 --n_epsilons 20
```

## Datasets

**Adult Income** (`adult`)
- Sensitive: `sex`, `race`
- Label: income >50K
- Preprocessing: binarize race (White vs. non-White), encode categoricals

**COMPAS** (`compas`)
- Sensitive: `sex`, `race`
- Label: no recidivism (two-year window)
- Preprocessing: filter to selected features, binarize race

**FolkTables** (`folktables`)
- Sensitive: `SEX`, `RAC1P`
- Label: income > median (TARGET)
- Preprocessing: standardize features, clip norms to 3

**CelebA** (`celeba`)
- Sensitive: `Male`
- Label: `Smiling`
- Preprocessing: standardize, clip norms to 2

See `src/data_loading/loaders.py` for download scripts for external datasets.

## Metrics

**Utility:**
- Accuracy: fraction of correct predictions

**Fairness:**
- Equal Opportunity Difference (EOD): difference in TPR across groups
  - EOD = TPR(A=1) - TPR(A=0)
  - Perfect fairness when EOD = 0
- Average Odds Difference (AOD): average of TPR and FPR differences
  - AOD = 0.5 × [(TPR₁ - TPR₀) + (FPR₁ - FPR₀)]
- Demographic Parity: difference in positive prediction rates
  - SPD = PR(A=1) - PR(A=0)

**Privacy:**
- Differential Privacy (ε, δ): measured by the Laplace mechanism
  - Noise scale for Laplace: σ = Δf / ε where Δf is sensitivity
  - We enforce DP by perturbing joint probabilities P(A,Y) before reweighting
  - δ = 1/n² in our experiments

## Reproducing Results

**Step 1:** Download datasets (if needed)
```bash
cd fairnessprivacy && python folktables_download.py
```

**Step 2:** Run privacy-fairness sweep
```bash
python -m src.experiments.privacy_curve --dataset folktables --epsilon_min 0.001 --epsilon_max 1.0
```

**Step 3:** Compare mitigation strategies
```bash
python -m src.experiments.compare_mitigations --dataset adult --epsilon 0.5
```

**Step 4:** Generate plots
```bash
python src/utils/plot.py --input_dir results/fairness_fct_epsilon --output plots/
```

Results are saved to `/results` as timestamped CSV files and pickled models. Config hyperparameters can be overridden via CLI arguments.

## Citation

```bibtex
@misc{privacyfairness2024,
  author = {Your Name},
  title = {Privacy vs Fairness in Machine Learning},
  year = {2024},
  url = {https://github.com/yourusername/PrivacyFairness}
}
```

## Disclaimer

This codebase is research-oriented and should not be used for production compliance decisions without independent review. Metric implementations follow standard definitions but may contain bugs. Use at your own risk.

