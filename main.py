import time
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from apu_pipeline import (
    DATA_PATH,
    IST,
    MODEL_PATH,
    ROOT,
    build_feature_frame,
    fetch_open_meteo,
    hourly_to_30min,
    load_holidays,
    load_model,
    next_30min_blocks,
    predict_blocks,
)

CACHE_TTL_SECONDS = 300

app = FastAPI(
    title="APU Demand Forecasting API",
    description="24-hour electricity demand forecast for Apex Power & Utilities (Dhanbad, Jharkhand, India), served from a trained HistGradientBoosting model with Open-Meteo weather and localized Dhanbad holiday features.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_artifact = None
_cache = {"key": None, "ts": 0.0, "payload": None}


def get_artifact():
    global _artifact
    if _artifact is None:
        _artifact = load_model()
    return _artifact


def climatology_weather(blocks: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH / "weather_dhanbad.csv", parse_dates=["ts"])
    df["ts"] = df["ts"].dt.tz_localize(None)
    clim = df.set_index("ts").groupby([df["ts"].dt.month, df["ts"].dt.hour])[
        ["temperature", "humidity", "cloud_cover", "wind_speed"]
    ].mean()
    naive = blocks.tz_localize(None)
    out = pd.DataFrame([clim.loc[(ts.month, ts.hour)] for ts in naive], index=naive)
    out = out.reset_index().rename(columns={"index": "ts"}).round(3)
    out["ts"] = out["ts"].dt.tz_localize(IST)
    return out


def window_payload(n_blocks: int) -> dict:
    global _cache
    if _cache["key"] == n_blocks and time.time() - _cache["ts"] < CACHE_TTL_SECONDS:
        return _cache["payload"]

    artifact = get_artifact()
    now = datetime.now(IST)
    blocks = next_30min_blocks(now, n_blocks)
    start_d = blocks[0].strftime("%Y-%m-%d")
    end_d = (blocks[-1] + pd.Timedelta(hours=24)).strftime("%Y-%m-%d")

    weather_source = "Open-Meteo forecast API (live)"
    try:
        wx = hourly_to_30min(fetch_open_meteo(start_d, end_d, archive=False))
    except Exception:
        wx = climatology_weather(blocks)
        weather_source = "climatological fallback (monthly-hourly means from cached Open-Meteo archive)"

    holidays = load_holidays()["holidays"]
    X = build_feature_frame(blocks, wx, holidays)
    yhat = predict_blocks(artifact, X)

    wx_blocks = wx.set_index("ts").reindex(blocks)
    hol = pd.DataFrame({"date": pd.Series(blocks.tz_localize(None).date)})
    hol = hol.merge(pd.DataFrame(holidays), on="date", how="left")

    meta = {
        "location": "Dhanbad, Jharkhand, India (23.7957 N, 86.4304 E)",
        "generated_at_ist": now.strftime("%Y-%m-%d %H:%M"),
        "forecast_start_ist": str(blocks[0]),
        "forecast_end_ist": str(blocks[-1]),
        "n_blocks": int(len(blocks)),
        "block_minutes": 30,
        "model": artifact["metadata"]["model_type"],
        "holdout_metrics": artifact["metadata"]["holdout_metrics"],
        "units": "kW (average power per 30-min block)",
        "weather_source": weather_source,
    }
    forecast = [
        {
            "block": i,
            "timestamp_ist": str(ts),
            "label": ts.strftime("%d %b, %H:%M"),
            "predicted_demand_kw": round(float(p), 1),
            "holiday": None if pd.isna(nm) else nm,
            "holiday_category": None if pd.isna(nm) else cat,
            "holiday_impact": None if pd.isna(nm) else imp,
        }
        for i, (ts, p, nm, cat, imp) in enumerate(zip(
            blocks, yhat, hol["name"], hol["category"], hol["impact"]))
    ]
    weather = [
        {
            "block": i,
            "timestamp_ist": str(ts),
            "label": ts.strftime("%d %b, %H:%M"),
            "temperature_c": round(float(t), 1),
            "humidity_pct": round(float(hu), 1),
            "cloud_cover_pct": round(float(cl), 1),
            "wind_speed_ms": round(float(wd), 2),
        }
        for i, (ts, t, hu, cl, wd) in enumerate(zip(
            blocks,
            wx_blocks["temperature"], wx_blocks["humidity"],
            wx_blocks["cloud_cover"], wx_blocks["wind_speed"]))
    ]
    in_window = hol.dropna(subset=["name"]).drop_duplicates(subset=["date", "name"])
    holiday_list = [
        {
            "date": row["date"],
            "name": row["name"],
            "category": row["category"],
            "impact": row["impact"],
        }
        for _, row in in_window.iterrows()
    ]

    payload = {
        "meta": meta,
        "forecast": forecast,
        "weather": weather,
        "holidays": holiday_list,
        "holiday_categories": load_holidays()["categories"],
    }
    _cache = {"key": n_blocks, "ts": time.time(), "payload": payload}
    return payload


@app.get("/", include_in_schema=False)
def root():
    index_html = ROOT / "public" / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return {
        "service": "APU Demand Forecasting API",
        "endpoints": ["/api/health", "/api/forecast", "/api/weather", "/api/holidays", "/api/dashboard", "/docs"],
    }


@app.get("/api/health")
def health():
    m = get_artifact()["metadata"]
    return {
        "status": "ok",
        "model": m["model_type"],
        "target": m["target"],
        "trained_on": m["training_range"],
        "holdout_metrics": m["holdout_metrics"],
        "artifact_kb": round(MODEL_PATH.stat().st_size / 1024, 1),
    }


@app.get("/api/forecast")
def forecast(blocks: int = Query(default=48, ge=1, le=96)):
    try:
        payload = window_payload(blocks)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"forecast generation failed: {e}")
    return {"meta": payload["meta"], "forecast": payload["forecast"]}


@app.get("/api/weather")
def weather(blocks: int = Query(default=48, ge=1, le=96)):
    try:
        payload = window_payload(blocks)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"weather fetch failed: {e}")
    return {"meta": payload["meta"], "weather": payload["weather"]}


@app.get("/api/holidays")
def holidays(blocks: int = Query(default=48, ge=1, le=96)):
    payload = window_payload(blocks)
    return {
        "meta": payload["meta"],
        "holidays": payload["holidays"],
        "categories": payload["holiday_categories"],
    }


@app.get("/api/dashboard")
def dashboard(blocks: int = Query(default=48, ge=1, le=96)):
    try:
        return window_payload(blocks)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"dashboard payload failed: {e}")
