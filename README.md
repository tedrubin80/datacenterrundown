# Climate-Driven Datacenter Location Optimization

### Machine Learning Approaches to Dynamic TCO Modeling Under Global Warming Scenarios

**Ted Rubin** · Independent Researcher · ted@theorubin.com · March–April 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://zenodo.org)

---

## Known Issues & Reproducibility Notes

Before running the pipeline, read these three points:

**1. FEMA data must be downloaded separately.**
The `data/` directory is empty. The Weibull survival model concordance of 0.944 reported in the paper requires the real FEMA dataset. Without it, the pipeline falls back to synthetic data and produces a concordance of ~0.57. Download first:
```bash
make download-fema   # downloads ~35 MB from OpenFEMA API
```

**2. AUROC = 1.00 is a synthetic-data artifact.**
The outage classifier (Notebook 04) achieves AUROC = 1.00 on synthetic labels because the label generation function uses the same variables (n_events, severity, temperature) as the XGBoost features. This reflects internal consistency of the synthetic construction, not real-world predictive performance. When real outage records are substituted, AUROC will be lower. This is documented in Section 5.4 of the paper.

**3. Reference facility is 100 MW throughout.**
Both the facility TCO (Monte Carlo) and hardware TCO use a 100 MW / 500-rack reference facility, consistent with DCCore §2.1 and the Hardware Cost Addendum.

---

## Overview

This repository contains all research software, configuration, notebooks, and paper source for the study of climate-driven datacenter economics. It extends **DCCore (Rubin, 2025)** — a Monte Carlo analysis of global datacenter location strategies — by introducing two machine learning contributions:

1. **Climate-Dynamic TCO Model** — Ensemble (XGBoost + Random Forest) and Bayesian Neural Network models replace static cost distributions with year-by-year predictions that shift based on IPCC RCP climate pathways (RCP 2.6, 4.5, 8.5).

2. **Extreme Weather Impact Model** — Weibull AFT survival analysis, XGBoost outage classification, and quantile gradient boosted regression for insurance premium trajectories, calibrated on 69,769 real FEMA disaster records (1953–2026).

### Key Findings

| Finding | Value |
|---|---|
| Nordic vs. Atlanta TCO gap (static) | ~74% |
| Nordic vs. Atlanta TCO gap (RCP 8.5, 2050) | >80% |
| Hardware energy vs. facility TCO ratio (AI racks) | 5–10× |
| Boden–Atlanta combined 25-year gap | $2.1B |
| Atlanta climate exposure vs. Boden (RCP 8.5) | 4.6× |
| Weibull survival model concordance index | 0.944 |
| Georgia vs. Wyoming disaster declarations/yr | 148.7 vs. 9.0 (16×) |

---

## Repository Structure

```
datacenterrundown/
│
├── configs/                        # All model and scenario configuration
│   ├── locations.yaml              # 10 global market profiles (DCCore parameters)
│   ├── climate_scenarios.yaml      # IPCC RCP 2.6 / 4.5 / 8.5 definitions
│   ├── model_params.yaml           # BNN, XGBoost, survival model hyperparameters
│   ├── eia_calibration.yaml        # EIA electricity price calibration
│   └── syngen/                     # Synthetic data generation schemas
│       ├── climate_variables.json
│       ├── extreme_events.json
│       ├── financial_variables.json
│       └── insurance_variables.json
│
├── notebooks/                      # Jupyter notebooks (clean + executed versions)
│   ├── 01_eda_paper_replication    # Monte Carlo replication of DCCore baseline
│   ├── 02_real_data_exploration    # EIA, NOAA, FEMA API data exploration
│   ├── 03_idea3_dynamic_tco        # Climate-dynamic TCO model (physics + BNN)
│   ├── 04_idea5_weather_insurance  # Survival analysis, outage classifier, insurance
│   ├── 05_results_synthesis        # Combined framework results and figures
│   └── 06_bnn_uncertainty          # Bayesian uncertainty calibration analysis
│
├── src/                            # Python source package
│   ├── data/                       # Data loading, APIs, synthetic generation
│   │   ├── location_profiles.py    # Location config loader (LocationProfile dataclass)
│   │   ├── climate_projections.py  # Year-by-year RCP projection generator
│   │   ├── dataset.py              # PyTorch Dataset for BNN training
│   │   ├── load_eia.py / pull_eia.py     # EIA electricity price API
│   │   ├── load_fema.py / pull_fema.py   # OpenFEMA disaster declarations API
│   │   ├── pull_noaa.py            # NOAA CDO climate data API
│   │   ├── syngen_runner.py        # SynGen + Gaussian copula data generator
│   │   └── correlation_engine.py   # Cross-variable correlation injection
│   │
│   ├── tco/                        # TCO computation layer
│   │   ├── components.py           # TCOParams dataclass, compute_tco()
│   │   ├── monte_carlo.py          # Static Monte Carlo (10,000 iterations)
│   │   ├── dynamic_distributions.py # Climate-shifted distributions (100,000 iter.)
│   │   ├── hardware_costs.py       # Hardware tier TCO (H100, PC, Hybrid, ARM)
│   │   └── discount.py             # Discounting utilities (r = 0.07)
│   │
│   ├── models/
│   │   ├── idea3/                  # Dynamic TCO models
│   │   │   ├── bayesian_nn.py      # BayesianTCONet (MC Dropout, 3 layers)
│   │   │   ├── ensemble_model.py   # XGBoost + Random Forest ensemble
│   │   │   └── trainer.py          # Training loop with early stopping
│   │   └── idea5/                  # Extreme weather models
│   │       ├── survival_model.py   # Weibull AFT (lifelines)
│   │       ├── event_classifier.py # XGBoost with SHAP explanations
│   │       ├── insurance_regressor.py # Quantile GBM (5 quantiles)
│   │       └── trainer.py          # Unified training and evaluation
│   │
│   ├── risk/
│   │   ├── metrics.py              # CV, VaR, CVaR, risk_premium()
│   │   └── scenario_comparator.py  # Cross-scenario ranking and comparison tables
│   │
│   ├── visualization/
│   │   ├── tco_plots.py            # TCO distribution and comparison charts
│   │   ├── climate_plots.py        # RCP trajectory and PUE degradation plots
│   │   ├── risk_heatmaps.py        # Location × scenario risk heatmaps
│   │   └── survival_plots.py       # Kaplan-Meier and hazard rate plots
│   │
│   └── utils/
│       └── constants.py            # Shared constants (HOURS_PER_YEAR, etc.)
│
├── pipelines/                      # End-to-end runner scripts
│   ├── full_pipeline.py            # Runs all phases in sequence
│   ├── generate_data.py            # Data generation phase
│   ├── train_idea3.py              # Dynamic TCO model training
│   ├── train_idea5.py              # Weather/insurance model training
│   ├── train_bnn.py                # BNN-only training
│   ├── run_combined_tco.py         # Combined facility + hardware TCO
│   └── eia_comparison.py           # EIA price calibration comparison
│
├── tests/                          # Pytest test suite
│   ├── conftest.py                 # Shared fixtures (Boden profile, sample data)
│   ├── test_tco/test_components.py # TCO computation unit tests
│   └── test_risk/test_metrics.py   # Risk metrics unit tests
│
├── paper/
│   └── climate_driven_datacenter_tco.tex  # LaTeX source for the paper
│
├── public/                         # Static web dashboard (Railway deployment)
│   ├── index.html / style.css
│   ├── server.py                   # Minimal Python HTTP server
│   └── Dockerfile / railway.json
│
├── DCCore.pdf                              # Predecessor study (Rubin, 2025)
├── DCCore_PC_Hardware_Cost_Addendum.pdf    # Hardware cost projections addendum
├── ARM_TCO.pdf               # ARM efficiency tier analysis
│
├── requirements.txt
├── Makefile
├── LICENSE                         # MIT (code) / CC BY 4.0 (data & papers)
├── CITATION.cff
└── .zenodo.json
```

---

## Locations Analyzed

| # | Location | Tier | Currency | PUE | Power (mode) | 10Y Facility TCO |
|---|---|---|---|---|---|---|
| 1 | Boden, Sweden | Nordic | EUR | 1.09 | €22.5/MWh | €499.5M |
| 2 | Iceland (Reykjanes) | Nordic | EUR | 1.08 | €23.0/MWh | €582.4M |
| 3 | Kristiansand, Norway | Nordic | EUR | 1.10 | €25.0/MWh | €596.9M |
| 4 | Sines, Portugal | Emerging | EUR | 1.22 | €50.0/MWh | €679.5M |
| 5 | Evanston, Wyoming | US Secondary | USD | 1.08 | $28.5/MWh | $1,270.2M |
| 6 | Des Moines, Iowa | US Secondary | USD | 1.20 | $42.0/MWh | $1,612.8M |
| 7 | Salt Lake City, Utah | US Secondary | USD | 1.18 | $45.0/MWh | $1,619.1M |
| 8 | Florence, SC | US Secondary | USD | 1.25 | $55.0/MWh | $1,667.8M |
| 9 | Johor, Malaysia | Emerging | USD | 1.45 | $75.0/MWh | $1,681.8M |
| 10 | Atlanta, Georgia | Traditional | USD | 1.35 | $85.0/MWh | $1,952.1M |

---

## Quickstart

### Requirements
- Python 3.10+
- See `requirements.txt` for all dependencies

### Install

```bash
git clone https://github.com/theorubin/datacenterrundown
cd datacenterrundown
pip install -r requirements.txt
```

### Run full pipeline

```bash
# Step 1: Download real FEMA data (required for full Idea 5 concordance = 0.944)
make download-fema

# Step 2: Run full analysis
make all
# or explicitly:
python pipelines/full_pipeline.py --seed 42 --rows 10000
```

### Run individual phases

```bash
make download-fema  # Download FEMA disaster declarations (~35 MB)
make data           # Generate synthetic + real data
make idea3          # Train dynamic TCO models
make idea5          # Train weather/insurance models
make test           # Run test suite
```

### Use notebooks

The `notebooks/` directory contains both clean (`.ipynb`) and pre-executed (`_executed.ipynb`) versions of all six notebooks. Run them in order (01 → 06) for the full analysis.

```bash
jupyter lab notebooks/
```

---

## Data Sources

### Real Data (via API)
| Source | Data | Coverage |
|---|---|---|
| [EIA API](https://www.eia.gov/opendata/) | US retail electricity prices by state | 2015–2025 |
| [NOAA CDO API](https://www.ncdc.noaa.gov/cdo-web/) | Daily temperature, precipitation, wind | 2015–2024 |
| [OpenFEMA API](https://www.fema.gov/about/openfema) | Disaster declarations, damage assessments | 2000–2026 (69,769 records) |

### Synthetic Data
International location data is generated using the SynGen framework with a Gaussian copula post-processor to inject realistic correlations between climate and financial variables. Schema definitions are in `configs/syngen/`.

### Configuration Data
All 10 location profiles are parameterized in `configs/locations.yaml` with triangular distribution bounds (min, mode, max) sourced from DCCore (2025).

---

## Climate Scenarios

Three IPCC Representative Concentration Pathways modeled from 2025 to 2050:

| Scenario | Description | ΔT by 2050 | Extreme Event Multiplier |
|---|---|---|---|
| RCP 2.6 | Aggressive mitigation | +1.5°C | 1.15× |
| RCP 4.5 | Moderate mitigation | +2.1°C | 1.50× |
| RCP 8.5 | Business as usual | +3.0°C | 2.25× |

---

## Hardware Tiers (100 MW / 500-rack reference facility)

| Tier | Rack Cost | Power/Rack | Refresh | Maint | 10Y HW TCO (Boden) |
|---|---|---|---|---|---|
| Standard AI (H100) | $3.5M | 100 kW | 4 yr | 12% | €9.29B |
| Traditional PC | $500K | 20 kW | 5 yr | 10% | €1.13B |
| Hybrid GPU+CPU | $2.0M | 60 kW | 4 yr | 11% | €5.20B |
| ARM Efficiency | $800K | 15 kW | 5 yr | 8% | ~€1.1B |

---

## Model Architecture

### Idea 3: Dynamic TCO (Climate-Shifted Monte Carlo)

Physics-based shift functions (from Rubin 2026, §3.4):

```
ΔPUE(t)          = ΔT(t) × β_PUE × (1 + H_f) × (1 + 0.05 × ΔT(t))
Δpower(t)        = P_base × (δprice(t) + 0.3 × δCDD(t)) / 100
σ_insurance(t)   = (E(t) / E₀)^1.5
```

Validated by Bayesian Neural Network (MC Dropout, 50 samples):
- Architecture: 3 hidden layers (128 → 64 → 32), dropout = 0.1
- Training: 12,483 parameters, early stopping at epoch 193
- Calibration: 83–93% within 1σ, 94–100% within 2σ

### Idea 5: Extreme Weather Impact

- **Survival model**: Weibull AFT (`lifelines`), concordance index = 0.944
- **Outage classifier**: XGBoost with SHAP, AUROC = 1.00
- **Insurance regressor**: Quantile GBM at [5, 25, 50, 75, 95] percentiles, RMSE = $0.71M

---

## Citation

If you use this work, please cite:

```bibtex
@software{rubin2026datacenter,
  author    = {Rubin, Ted},
  title     = {Climate-Driven Datacenter Location Optimization:
               Machine Learning Approaches to Dynamic TCO Modeling
               Under Global Warming Scenarios},
  year      = {2026},
  month     = {April},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

Or see `CITATION.cff` for the full citation file.

**Predecessor study:**
```bibtex
@techreport{rubin2025dccore,
  author = {Rubin, Ted},
  title  = {Global Datacenter Location Optimization:
            A Monte Carlo Simulation Analysis of Strategic Markets},
  year   = {2025},
  note   = {DCCore Research Series}
}
```

---

## Related Work

- **DCCore.pdf** — The predecessor Monte Carlo analysis (included in this repo)
- **DCCore_PC_Hardware_Cost_Addendum.pdf** — Year-by-year hardware cost projections for all 10 locations across 3 hardware tiers
- **ARM_TCO.pdf** — ARM efficiency tier deep-dive analysis

---

## License

Code: MIT License — see [LICENSE](LICENSE)

Data and papers: Creative Commons Attribution 4.0 International (CC BY 4.0)
