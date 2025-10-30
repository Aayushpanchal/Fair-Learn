# Proposed Repository Structure

```
PrivacyFairness/
├── README.md                          # Professional research-grade README
├── requirements.txt                   # Pinned dependencies
├── .gitignore                         # Python project gitignore
├── REPO_STRUCTURE.md                  # This file
│
├── src/                               # Main source code
│   ├── __init__.py
│   ├── data_loading/                  # Dataset loaders
│   │   ├── __init__.py
│   │   └── loaders.py                 # All dataset load functions
│   ├── models/                        # Model definitions
│   │   ├── __init__.py
│   │   └── trainer.py                 # Training logic with DP and mitigation
│   ├── metrics/                       # Fairness and privacy metrics
│   │   ├── __init__.py
│   │   ├── fairness_metrics.py        # EOD, AOD, demographic parity
│   │   └── privacy_metrics.py         # DP parameters, noise mechanisms
│   ├── experiments/                   # Experiment runners
│   │   ├── __init__.py
│   │   ├── train.py                   # Main CLI training script
│   │   └── privacy_curve.py           # (TODO: sweep epsilon values)
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── config.py                  # Centralized config dataclasses
│       ├── logging.py                 # Logging setup
│       └── seeding.py                 # Reproducibility seeds
│
├── notebooks/                         # Exploratory notebooks (archive)
│   ├── privacy_curve_equalOpportunity.ipynb
│   ├── privacy_curve_equalOdds.ipynb
│   └── ...
│
├── results/                           # Experimental outputs
│   ├── checkpoints/                   # Saved models
│   └── fairness_fct_epsilon/          # Privacy-fairness curve data
│
├── data/                              # Datasets (gitignored if large)
│   ├── adult.csv
│   ├── compas-scores-two-years.csv
│   ├── data.csv (folktables)
│   └── ...
│
└── fairnessprivacy/                   # OLD CODE (deprecated)
    ├── README.md
    ├── expe_fairness_fct_epsilon.py
    ├── load_dataset.py
    └── ...                            # Will be removed after migration
```

## Migration Status

### ✅ Completed
- Professional README.md
- Modular src/ structure with clear separation
- Centralized config.py with dataclasses
- CLI entrypoint in src/experiments/train.py
- Fairness metrics module with docstrings
- Privacy metrics module
- Trainer with reweighting mitigation
- Requirements.txt and .gitignore

### 🔄 To Complete
- [x] Fix imports in train.py (adjust relative imports) 
- [x] Implement privacy_curve.py experiment for epsilon sweeps
- [ ] Migrate remaining useful code from fairnessprivacy/ (compute_bounds.py for theoretical bounds)
- [ ] Archive or delete deprecated fairnessprivacy/ directory after migration
- [x] Add data/.gitkeep to preserve empty data/ in git
- [ ] Review notebooks: They study sample size vs fairness, not DP vs fairness (different research question)

## Next Steps

1. Fix import paths in src/experiments/train.py (use absolute imports)
2. Test the training script with one dataset
3. Migrate the privacy curve experiment logic
4. Archive old notebooks
5. Clean up

