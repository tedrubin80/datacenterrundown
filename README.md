# AI Infrastructure Economics: A Complete Decision Framework for Data Center Location, Hardware Selection, and Climate Risk

Machine learning approaches to dynamic total cost of ownership modeling under global warming scenarios for AI datacenter infrastructure across 10 global markets.

**[Read the Research Paper](https://theorubin.com/?page=research&slug=ai-infrastructure-economics-a-complete-decision-framework-for-data-center-location-hardware-selection-and-climate-risk-2025-2026-research-series)** | **[Kaggle Dataset](https://www.kaggle.com/datasets/theodorerubin/climate-driven-datacenter-tco)** | **[Project Homepage](https://datacenterrundown.com)**

## Key Findings

- **$2.1B** 25-year TCO gap between Boden, Sweden and Atlanta, Georgia
- **$179M** climate premium for Atlanta under RCP 8.5
- **4.6x** climate exposure ratio between Nordic and traditional hub locations
- **0.94** concordance index for survival model using real FEMA disaster data
- **85%** energy reduction with ARM-based hardware, saving $131-491M per facility

## Repository Structure

```
.
├── configs/                  # YAML configurations for locations, climate scenarios, models
├── data/
│   ├── raw/                  # EIA, NOAA, FEMA source data
│   ├── processed/            # Cleaned and correlated datasets
│   ├── results/              # Model outputs
│   └── splits/               # Train/test splits
├── notebooks/                # 6 Jupyter notebooks (analysis + executed versions)
│   ├── 01_eda_paper_replication
│   ├── 02_real_data_exploration
│   ├── 03_idea3_dynamic_tco
│   ├── 04_idea5_weather_insurance
│   ├── 05_results_synthesis
│   └── 06_bnn_uncertainty
├── paper/                    # LaTeX manuscript
├── pipelines/                # Runnable training and data generation scripts
├── public/                   # Project homepage
├── src/                      # Core library (models, risk, tco, data, visualization)
├── tests/                    # pytest test suite
├── Makefile                  # Build targets
└── requirements.txt          # Python dependencies
```

## Methodology

| Component | Technique | Key Metric |
|-----------|-----------|------------|
| Dynamic TCO | Monte Carlo simulation with physics-based climate shift functions (100K+ iterations) | PUE degrades at 0.008-0.012 per degree C |
| Survival Analysis | Weibull AFT model for time-to-outage prediction | Concordance: 0.94 |
| Insurance Modeling | Quantile Gradient Boosted regression with nonlinear event scaling | RMSE: $0.71M |
| Validation | Bayesian Neural Network (MC Dropout, 12K params) | 94-100% calibration coverage |

## Data Sources

- **EIA API** - US retail electricity prices and generation mix (2015-2025)
- **NOAA CDO API** - Historical daily temperature, precipitation, wind (46,044 records)
- **FEMA OpenData** - 69,769 disaster declarations (1953-2026)
- **DCCore Paper** - Monte Carlo simulation parameters for 10 global markets

## Quickstart

```bash
# Install dependencies
make install

# Generate synthetic data
make data

# Run full pipeline (data + all models)
make all

# Run individual components
make idea3    # Dynamic TCO model
make idea5    # Weather/insurance model

# Run tests
make test
```

### Requirements

- Python 3.10+
- numpy, pandas, scipy, scikit-learn, matplotlib, seaborn
- xgboost, lifelines, shap, torch

See [requirements.txt](requirements.txt) for full list.

## Hardware Tiers

| Tier | Example | Cost/Rack | Power | Refresh |
|------|---------|-----------|-------|---------|
| Standard AI | H100 GPU | $3.5M | 100 kW | 4 yr |
| Hybrid GPU+CPU | H100 PCIe + CPU | $2.0M | 60 kW | 4 yr |
| ARM Efficiency | Apple Silicon | $800K | 15 kW | 5 yr |
| Traditional PC | Enterprise CPU | $500K | 20 kW | 5 yr |

## Citation

If you use this work, please cite:

```bibtex
@article{rubin2026climate,
  title={AI Infrastructure Economics: A Complete Decision Framework for Data Center Location, Hardware Selection, and Climate Risk},
  author={Rubin, Ted},
  year={2026},
  url={https://theorubin.com/?page=research&slug=ai-infrastructure-economics-a-complete-decision-framework-for-data-center-location-hardware-selection-and-climate-risk-2025-2026-research-series}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

Dataset on Kaggle is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
