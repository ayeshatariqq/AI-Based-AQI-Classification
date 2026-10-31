# AI-Based AQI Classification

A machine-learning pipeline and desktop application that classify the **air-quality category** of a city-day (*Good, Satisfactory, Moderate, Poor, Very Poor, Severe*, following India's CPCB scheme) from daily pollutant readings.

Built during an eight-week internship project at ASCEND Tower (IT and Software Development Lab). The repository holds the whole workflow: data cleaning, exploratory analysis, baselines, ensembles, class-imbalance handling, tuning, calibration, a desktop application, and a generalization study on two unseen datasets.

## Highlights

| | Result |
|---|---|
| Final model | XGBoost + SMOTE, hyperparameter-tuned, probability-calibrated |
| Held-out test set (2,415 city-days, 20 cities) | **82.7% accuracy, macro F1 0.791** |
| Unseen Indian city (Pune, 276 days) | 82.6% accuracy\* |
| Unseen country (Beijing, 1,453 days) | 71.4% accuracy\*: 83% on days not driven by ozone, 56% on ozone-driven days |

\* Six-pollutant sibling of the final model, trained with CO and O3 as 8-hour maxima. See [Generalization](#generalization).

## Repository structure

```
.
├── README.md
├── requirements.txt            # notebook dependencies
├── model_training.ipynb        # cleaning -> EDA -> baselines -> ensembles -> SMOTE -> tuning -> calibration -> test
├── generalization.ipynb        # Pune and Beijing study
├── data/                       # input data (see "Data" below)
├── models/                     # saved artifacts written by model_training.ipynb
├── aether/                     # final desktop application (PySide6)
├── prototype/
│   └── app_tkinter.py          # first prototype (Tkinter)
└── docs/                       # full report and short summary
```

## Quick start

**1. Environment** (Python 3.12 recommended)

```bash
pip install -r requirements.txt
```

**2. Reproduce the model.** Open `model_training.ipynb` and run it top to bottom. It needs only `data/city_day.csv` and writes the saved artifacts to `models/`:

| File | Contents |
|---|---|
| `xgb_aqi_model.pkl` | the calibrated model |
| `label_encoder.pkl` | class names <-> integers |
| `feature_columns.pkl` | the 31 training columns, in order |
| `pollutant_columns.pkl` | the eight pollutant inputs |
| `city_list.pkl`, `season_list.pkl` | supported cities and seasons |

Random seeds are fixed (42) wherever randomness is involved.

**3. Run the desktop application**

```bash
cd aether
pip install -r requirements.txt
# copy the .pkl files from ../models into aether/models
python app.py
```

A standalone Windows executable can be built with `build_exe.bat` (see `aether/README.md`).

**4. Reproduce the generalization study.** Add the extra data below, then run `generalization.ipynb` top to bottom. It reads only the raw files.

## Data

| Dataset | Used for | Where to get it |
|---|---|---|
| *Air Quality Data in India (2015-2020)*, Rao (Kaggle) | Training. `city_day.csv` for the model; `city_hour.csv` for the generalization study | https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india |
| Daily AQI and pollutant concentrations for Pune (Mendeley Data) | Generalization (new Indian city). Place `city_averaged.csv` in `data/pune/` | https://data.mendeley.com/datasets/y6p778djzj/1 |
| Beijing Multi-Site Air-Quality Data (UCI) | Generalization (new country). Place the 12 `PRSA_Data_*.csv` files in `data/beijing/` | https://archive.ics.uci.edu/dataset/501/beijingmultisiteairqualitydata |

`data/city_day.csv` is already in this repository. Each dataset has its own licence; check its source page before redistributing.

## Approach

| Step | Decision | Why |
|---|---|---|
| Dataset | India city-level daily data | Real, labelled, multi-city, precedented in the literature |
| Missing PM10 | Drop the six cities where it is structurally absent | PM10 is a core AQI pollutant; filling years of absent data would be inventing it |
| Other gaps | Forward/back-fill within each city | Pollution is autocorrelated; avoids leakage between cities |
| Outliers | Kept | They are real severe-pollution days |
| Split | Chronological 70/15/15 **within each city** | No future data in training; all 20 cities in every set |
| Metric | Macro F1 and per-class recall | Accuracy hides failures on rare, important classes |
| Imbalance | SMOTE on training data only (inside CV folds when tuning) | Severe recall rose from 0.79 to 0.94 |
| Calibration | Isotonic, pre-fitted on the validation set | Honest confidence without changing predictions |

### Results

| Stage | Macro F1 | Set |
|---|---|---|
| Logistic Regression (best baseline) | 0.692 | validation |
| XGBoost | 0.782 | validation |
| + SMOTE | 0.787 | validation |
| + hyperparameter search | 0.806 | validation |
| + calibration (final) | **0.791** (accuracy 0.827) | **test, evaluated once** |

## Desktop application

**Aether** classifies a single reading or a whole CSV file and says how confident it is.

- **Single reading:** eight pollutant readings, a city and a month; shows the category, calibrated confidence, plain-language guidance and the probability of every category, with a notice when a result is borderline.
- **Batch analysis:** open or drag in a CSV; sortable, filterable results; export. Rows that cannot be classified are skipped with a stated reason (unsupported city, invalid date, missing or negative reading).
- **History** of past single readings, and a **model card** with limitations.
- One prediction engine mirrors the training preprocessing exactly, so every view classifies identically.

Inputs are **daily averages**; CO is in mg/m3 and all other pollutants in ug/m3.

## Generalization

The test set shares cities and years with the training data, so the model was also checked on a new Indian city (Pune) and on Beijing, using a six-pollutant sibling of the final model.

| Test set | Days | CO/O3 as daily means | CO/O3 as 8-hour maxima |
|---|---|---|---|
| India test (20 cities) | 2,415 | 0.809 | 0.821 |
| Pune (unseen city) | 276 | 0.725 | 0.826 |
| Beijing (unseen country) | 1,453 | 0.641 | 0.714 |
| Beijing, days not set by ozone | 814 | not computed | 0.833 |
| Beijing, days set by ozone | 639 | not computed | 0.563 |

Findings:

1. **A definition mismatch, not model quality, caused the first failure.** The Indian data store CO and O3 as daily means; Pune and the official index use daily maxima of 8-hour averages. Rebuilding the training features the same way lifted Pune accuracy from 72.5% to 82.6%.
2. **The model is weakest where its training data are thin.** Only 14 of 11,227 training days have ozone driving a Poor-or-worse label, against 193 of Beijing's 1,453.
3. **Beijing's labels are computed**, with a CPCB-style calculator validated on Pune, not official categories.

## Limitations

- One training source: Indian monitoring data from 2015-2020. The city indicators tie the shipped model to 20 named cities.
- The shipped model was not directly tested on external data; the generalization results are for its six-pollutant sibling.
- Inputs must be daily averages; rare classes are small (27 Severe test days), so their metrics are noisy.
- Not a health advisory.

## Author

Ayesha Tariq, IT and Software Development Lab, ASCEND Tower. Supervisor: Raza Ali.

## Licence

Not yet specified. Add a `LICENSE` file before sharing or reusing this code.
