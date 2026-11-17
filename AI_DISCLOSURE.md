# AI-Use Disclosure

**Project:** Predicting Engine Failure from Sensor History and Comparing it with Fixed Replacement.  
**Course:** ES 335: Machine Learning (2026)  
**Authors:** Dhvanil Shah (`25110105`), Aarav Malde (`25110003`)  
**AI Assistant Used:** **Antigravity** (Google DeepMind's agentic AI coding assistant)

---

## 1. Summary of AI Tool Usage

During the development of this project and its accompanying static web report (`docs/`), **Antigravity** was used as an agentic pair-programming, front-end engineering, and documentation assistant. All modeling decisions, experimental protocols, hypothesis tests, and result interpretations were designed, directed, reviewed, and locally executed by the project authors.

---

## 2. Where Antigravity Was Used (Component Breakdown)

### A. Pipeline Engineering & Causality Verification (`src/`, `scripts/`)
- Structured modular `scikit-learn` `Pipeline` transformers in `src/features.py` (`SensorVarianceFilter`, `TrailingWindowFeatureTransformer`, `DataFrameStandardScaler`) to enforce fold-isolated sensor variance screening and strictly causal trailing windows.
- Assisted in writing automated causality and invariance checks in `scripts/run_features.py` to assert zero future-cycle look-ahead, cross-engine isolation, and row-order invariance.
- Organized deterministic CLI entry points in `scripts/` (`verify_data.py`, `make_splits.py`, `run_eda.py`, `run_features.py`, `run_cv.py`, `run_test_evaluation.py`, `export_web_data.py`).

### B. Static Web Report & Interactive Visualizations (`docs/`)
- Built the multi-section static web report in plain HTML, CSS, and ES-module JavaScript (`docs/index.html`, `docs/css/report.css`, `docs/js/main.js`) with zero external UI frameworks or charting CDNs.
- Implemented all client-side SVG charts (the Engine #24 hero trace and Figures 1–9), fallback figure wiring, progressive-disclosure drawers, and collapsible summary tables (Tables 3, 5, 6, and 8).
- Implemented the three live interactive widgets driven by `docs/data/report_data.json`:
  1. **Maintenance Cost & Threshold Explorer** (Figure 5 in Section 6: policy model selector, cost-ratio slider, alarm threshold slider, live metrics strip, and dual cost/outcome plots).
  2. **Engine-Level vs. Row-Split Leakage Comparison** (Figure 6 in Section 6: proper vs. random row split toggle and Engine #21 leakage inspection).
  3. **Held-Out Test Error & Trajectory Browser** (Figure 9 in Section 7: case-study selector and 20-engine audit browser).

### C. UX, Layout & Editorial Refinement
- Iterated on responsive desktop and mobile CSS grid alignment, typographic hierarchy (Literata + Public Sans), section scroll reveals (`IntersectionObserver`), bold formatting for key quantitative values, and automated headless-browser checks verifying zero horizontal overflow and zero overlapping SVG text labels.

---

## 3. Author Ownership & Quantitative Verification

### Experimental Design & Decisions
- The authors defined the core research framing (comparing strictly causal trailing sensor monitoring against optimal fixed-schedule replacement under asymmetric failure-to-waste costs), the deterministic **80/20** engine-level split (`seed=42`), the **5-fold** `GroupKFold` cross-validation protocol, the **6-rung** complexity ladder (Rungs 0–5), and all study limitations.

### Deterministic Local Execution & Data Audit
- Every pipeline script was executed locally against the official NASA C-MAPSS FD001 dataset (`CMAPSSData/`):
  - `python scripts/verify_data.py`
  - `python scripts/make_splits.py`
  - `python scripts/run_eda.py`
  - `python scripts/run_features.py`
  - `python scripts/run_cv.py`
  - `python scripts/run_test_evaluation.py`
  - `python scripts/export_web_data.py`
- Every split manifest (`data/splits/*.json`), cross-validation and held-out test metric (`results/metrics/*.csv`), locked development decision cutoff (*K*<sup>&ast;</sup> = 179, *p*<sup>&ast;</sup> = 0.99, &tau;<sup>&ast;</sup> = 9.0), cost sweep, and **1,000-engine** whole-engine bootstrap confidence interval reported in `README.md` and `docs/` traces directly to locally executed code and saved outputs in `results/metrics/` and `data/processed/test_evaluation_summary.json`. Zero quantitative metrics were estimated or fabricated by AI.
