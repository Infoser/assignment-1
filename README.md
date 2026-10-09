# Intelligent Power Demand Forecasting — Apex Power & Utilities (APU)

End-to-end prototype that forecasts electricity demand for every **30-minute block of the day (48 blocks)** for APU, a power provider serving **Dhanbad, Jharkhand, India**. Built for the Data Developer Intern assignment at Exascale Deeptech & AI Pvt. Ltd.

The trained model is served through a FastAPI backend and visualized on a single-page dashboard. The whole stack deploys to Docker **and** Vercel.

---

## 1. Repository structure

```
assignment-1/
├── eda.py                     Milestone 1 — EDA, data quality audit, cleaning, API weather integration
├── train_model.py             Milestones 2-3 — feature engineering, model comparison, training, artifact
├── apu_forecasting.ipynb      Jupyter notebook documenting the full analysis (EDA, cleaning, features, justification, model)
├── apu_pipeline.py            Shared core: feature contract, weather fetch, IST block logic, model I/O
├── main.py                    FastAPI backend (Vercel entrypoint + local/Docker server)
├── holidays_data.py           Self-sourced Dhanbad holiday list (generates data/holidays_dhanbad.json)
├── public/
│   └── index.html             Frontend dashboard (Chart.js)
├── data/
│   ├── Utility_consumption.csv       Provided mock data (10-min load, 3 feeders + weather)
│   ├── holidays_dhanbad.json         Generated localized holiday dataset (2017 + 2026)
│   ├── weather_dhanbad.csv           Cached Open-Meteo archive weather for Dhanbad, 2017
│   └── apu_30min_clean.csv           Cleaned 30-min dataset (generated)
├── models/
│   ├── apu_demand_model.pkl          Trained HistGradientBoosting artifact (~370 KB)
│   └── model_metadata.json           Feature contract, metrics, training range
├── figures/                          EDA + model figures (generated)
├── requirements.txt                  Runtime dependencies (API/serverless)
├── requirements-dev.txt              + matplotlib/seaborn/jupyter (analysis scripts)
├── vercel.json                       Vercel function configuration
├── Dockerfile                        Container deployment
└── .dockerignore
```

## 2. Quick start

### Run the API + dashboard locally (Python)

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 for the dashboard, or http://localhost:8000/docs for the interactive API docs.

### Run with Docker

```bash
docker build -t apu-forecast .
docker run -p 8000:8000 apu-forecast
```

### Deploy to Vercel

The project is Vercel-ready: `main.py` exposes a FastAPI `app` (a supported entrypoint), the dashboard lives in `public/` (served from the CDN), and `vercel.json` pins the Python version, function duration and the extra files the function needs (`models/`, `data/`, `apu_pipeline.py`).

1. Push this repository to GitHub.
2. In Vercel, **Add New Project → Import** the repo. Zero configuration is needed — Vercel auto-detects FastAPI.
3. Deploy. The dashboard is at `/`, the API at `/api/*`, docs at `/docs`.

Or with the CLI: `npm i -g vercel && vercel --prod`.

## 3. How the pipeline runs (reproducibility)

**Option A — Jupyter notebook (full documented walkthrough):**

```bash
pip install -r requirements-dev.txt
jupyter lab apu_forecasting.ipynb   # Run All — EDA, cleaning, features, model comparison, training, artifact
```

**Option B — headless scripts:**

```bash
pip install -r requirements-dev.txt
python eda.py            # Milestone 1: EDA prints + figures/01-07, cleans data -> data/apu_30min_clean.csv
python train_model.py    # Milestones 2-3: model comparison + figures/08-09, trains -> models/apu_demand_model.pkl
```

The notebook and the scripts run the same deterministic pipeline (fixed random seeds) and reuse the cached API weather, so they run offline after the first fetch.

## 4. Data sources

| Source | What | How |
|---|---|---|
| `Utility_consumption.csv` (provided) | 10-minute load for three 132 kV feeders (F1, F2, F3) + temperature/humidity/wind, Jan 1 – Dec 30, 2017 | Aggregated to 30-min block averages; system demand = F1+F2+F3 |
| [Open-Meteo](https://open-meteo.com/) public API | Temperature, humidity, **cloud cover**, wind speed for Dhanbad (23.7957° N, 86.4304° E) | Archive API for the training year (cached in `data/weather_dhanbad.csv`); forecast API live at request time for serving. No API key required |
| Self-sourced localized holidays | Jharkhand/Dhanbad festive + industrial holidays (Chhath, Tusu, Sarhul, Karma, Sohrai, Vishwakarma Puja, Durga Puja, Diwali, Eid, national days) | Compiled in `holidays_data.py` from the Jharkhand Government holiday list and regional festival calendars, with category (national/regional/industrial) and impact (high/medium/low). The 2017 list drives training; the 2026 list serves the live forecast window |

## 5. Key EDA findings (Milestone 1)

- **The provided CSV mixes two datetime formats** — `MM-DD-YYYY` with leading zeros for days 1–12 of each month and `M/D/YYYY` for days 13–31. A naive `dayfirst=True` parse silently misreads dates or loses 32% of rows to `NaT`. Per-element parsing (`format="mixed"`) recovers a **fully contiguous year**: 52,416 rows = 364 days × 144 slots, no gaps, no duplicates.
- **Demand shape**: single dominant evening peak (~98 MW at 20:00), morning trough (~50 MW at 06:00), mild weekly cycle (Sunday ratio 0.94), monsoon seasonal swing (Jul–Aug peak).
- **Weather**: temperature is the strongest driver (r = +0.49 overall, +0.45 in afternoon hours — cooling load); humidity is negatively correlated via seasonal collinearity. The provided CSV's weather is **nearly uncorrelated with real Dhanbad weather** (temp r = 0.56 with large −6.6 °C bias; humidity r = 0.13; wind r = 0.01) — it is mock data, so the model is trained on the geographically-correct Open-Meteo API weather, which is also the only source available at inference time (train/serve consistency).
- **Holidays**: category-aware flags (national/regional/industrial) matter — e.g., Vishwakarma Puja (the Dhanbad coal belt's biggest industrial shutdown) shows a measurable system-level effect, and Chhath/Diwali reshape the dawn/dusk profile rather than just the level. A generic national calendar would miss Tusu, Sarhul, Karma and Vishwakarma Puja entirely.
- **Outliers/gaps handled**: transient deviations (>30% from a 6 h rolling median at the 10-min level — e.g., the Apr 20 F1 feeder-trip dip of ~−53% for 20 min) are repaired by local time interpolation **before** aggregation so block means stay unbiased. Repeated exact values (e.g., 47,598.33 kW in F3 on different days) are quantization artifacts of coarse metering — kept and noted. No negative/zero readings or stuck-sensor flatlines are present.

## 6. Feature engineering & model justification (Milestone 2)

**Feature contract (21 features, shared verbatim by training and serving — see `apu_pipeline.py`):**

- *Calendar*: `block_of_day`, `hour`, `minute`, `dayofweek`, `is_weekend`, `month`, `dayofyear`, cyclical `sin/cos` of block-of-day (period 48) and day-of-year (period 365.25)
- *Weather (Open-Meteo, Dhanbad)*: `temperature`, `humidity`, `wind_speed`, `cloud_cover`, `cooling_degree` = max(0, T − 24 °C)
- *Holidays (localized)*: `is_holiday`, `is_national_holiday`, `is_regional_holiday`, `is_industrial_holiday`, `is_day_before_holiday`

**Architecture comparison** (holdout = December 2017, 1,440 blocks; 3-fold expanding-window CV on train):

| Model | MAE (kW) | RMSE (kW) | MAPE % | R² |
|---|---|---|---|---|
| Seasonal naive (D-1) | 2090 | 3314 | 3.43 | 0.95 |
| Seasonal naive (D-7) | 1799 | 2642 | 2.91 | 0.97 |
| Linear Regression | 8950 | 10257 | 14.08 | 0.48 |
| Random Forest | 1993 | 2676 | 3.34 | 0.96 |
| **Hist Gradient Boosting (chosen)** | **1995** | **2626** | **3.35** | **0.97** |
| HistGB + lags (rejected for serving) | 1302 | 1918 | 2.12 | 0.98 |

**Why this architecture (data-driven, see `train_model.py` output):**

1. Strong deterministic structure in the load (fixed daily peak/trough, weekly and monsoon cycles) → cyclical calendar encodings capture it without midnight/year discontinuities.
2. Weather is a moderate, non-linear driver → tree ensembles learn threshold/interaction effects natively; linear regression underfits badly (MAPE 14.08% vs 3.35%).
3. Holidays are category-specific events with distinct shapes → binary category flags preserve their differences.
4. **Chosen: `HistGradientBoostingRegressor`** on calendar + weather + holiday features — matches Random Forest on the holdout while winning RMSE, R² and CV, trains in seconds, and its artifact is a few hundred KB — ideal for serverless cold-start and bundle limits.
5. The lag-augmented variant is more accurate (MAPE 2.12%) but needs the previous day/week of load at inference — a stateless serverless function cannot reliably provide that. The direct model is the deployment-safe choice, and the absence of trend/lag features is what lets the artifact generalize to arbitrary future dates (the serving year is 9 years after the training year): every input is cyclical or externally sourced for the forecast date itself.

## 7. API endpoints (Milestone 3)

| Endpoint | Description |
|---|---|
| `GET /api/health` | Model status, training range, holdout metrics |
| `GET /api/forecast?blocks=48` | Fresh 24-hour forecast: 48 (default, up to 96) 30-min blocks with predicted demand + holiday markers |
| `GET /api/weather?blocks=48` | Temperature, humidity, cloud cover and wind speed for every block of the forecast window |
| `GET /api/holidays?blocks=48` | Localized Dhanbad holidays inside the forecast window + category legend |
| `GET /api/dashboard?blocks=48` | Combined payload (forecast + weather + holidays) for the frontend |
| `GET /docs` | Interactive OpenAPI docs |

The backend loads `models/apu_demand_model.pkl` lazily (cached per warm instance), fetches Open-Meteo live at request time, caches the window payload for 5 minutes, and falls back to climatology (monthly-hourly means from the cached archive) if the upstream weather API fails.

## 8. Frontend (Milestone 4)

`public/index.html` — a single-page dashboard (Chart.js):

- Forecast chart: predicted demand per 30-min block with holiday blocks highlighted (red = festival, purple = industrial) and holiday details in the tooltip
- Weather overlay chart: temperature, humidity, cloud cover, wind speed (dual axes)
- Summary cards: peak/min demand, average temperature, humidity, cloud cover, wind
- Localized holiday table (date, name, category, impact) or a "no holidays in window" notice

## 9. Notes

- The target is **average power (kW) per 30-minute block**, summed across the three 132 kV feeders — the standard dispatch-level quantity for block scheduling.
- The brief's line "forecast for the next 24 hours (96 blocks)" conflicts with the core objective "48 blocks total for 30-min intervals"; the API defaults to 48 blocks and accepts `?blocks=` up to 96 to cover either reading.
- Lunar festival dates (e.g., Chhath, Karma, Eid) are approximate and marked in `data/holidays_dhanbad.json`; local bandhs are not listed as they cannot be reliably pre-sourced.
