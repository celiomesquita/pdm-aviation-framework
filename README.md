# PDM Aviation Framework

Python implementation of the two case studies presented in:

> **Phase-Conditioned RUL Estimation and PBIT Fault Classification: A Reproducible Framework for Condition-Based Aviation Maintenance**

---

## Repository Structure

```
pdm-aviation-framework/
├── cs1_rul/        # Case Study 1: Phase-conditioned RUL estimation (C-MAPSS FD001)
└── cs2_pbit/       # Case Study 2: PBIT fault classification
```

## Case Study 1 — RUL Estimation (CS1)

- Dataset: C-MAPSS FD001 (publicly available at the [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository))
- Method: PCA-based damage index for P-point detection; Gradient Boosting regression on post-degradation cycles only
- Key result: MAE = 8.55 cycles, R² = 0.863

## Case Study 2 — PBIT Fault Classification (CS2)

- Dataset: Power-On Built-In Test data (anonymised; included in `cs2_pbit/`)
- Method: Temporal feature engineering with rolling statistics; Gradient Boosting classifier with strategic oversampling
- Key result: >95% accuracy across six failure modes on natural test distribution

## Installation

```bash
pip install -r requirements.txt
```

## Reproducibility

All experiments use random seed 42. Engine-unit and serial-number splits are applied to prevent data leakage. See the paper for the full reproducibility protocol.

## License

MIT License.

## Authors

