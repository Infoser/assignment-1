import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
IST = ZoneInfo("Asia/Kolkata")
DHANBAD_LAT, DHANBAD_LON = 23.7957, 86.4304
MODEL_PATH = ROOT / "models" / "apu_demand_model.pkl"
HOLIDAYS_PATH = ROOT / "data" / "holidays_dhanbad.json"
DATA_PATH = ROOT / "data"
FIGS_PATH = ROOT / "figures"

TARGET = "system_demand_kw"
AGG_FREQ = "30min"
BLOCKS_PER_DAY = 48

FEATURE_COLUMNS = [
    "block_of_day", "hour", "minute", "dayofweek", "is_weekend",
    "month", "dayofyear", "sin_block", "cos_block", "sin_doy", "cos_doy",
    "temperature", "humidity", "wind_speed", "cloud_cover", "cooling_degree",
    "is_holiday", "is_national_holiday", "is_regional_holiday",
    "is_industrial_holiday", "is_day_before_holiday",
]


def load_holidays(path=HOLIDAYS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def holiday_flags(timestamps: pd.DatetimeIndex, holidays: list) -> pd.DataFrame:
    dates = pd.Series(timestamps.date)
    by_date = {}
    day_before = set()
    for h in holidays:
        d = pd.Timestamp(h["date"]).date()
        by_date.setdefault(d, []).append(h)
        day_before.add(d - timedelta(days=1))

    def has_cat(d, cat):
        return int(any(h["category"] == cat for h in by_date.get(d, [])))

    return pd.DataFrame({
        "is_holiday": [int(d in by_date) for d in dates],
        "is_national_holiday": [has_cat(d, "national") for d in dates],
        "is_regional_holiday": [has_cat(d, "regional") for d in dates],
        "is_industrial_holiday": [has_cat(d, "industrial") for d in dates],
        "is_day_before_holiday": [int(d in day_before) for d in dates],
    }, index=timestamps)


def build_feature_frame(timestamps: pd.DatetimeIndex, weather: pd.DataFrame,
                        holidays: list) -> pd.DataFrame:
    ts = pd.DatetimeIndex(timestamps)
    w = weather.copy()
    if "ts" in w.columns:
        w = w.set_index("ts")
    w = w.reindex(ts)

    hour = ts.hour
    minute = ts.minute
    block = hour * 2 + (minute // 30)
    dow = ts.dayofweek
    doy = ts.dayofyear
    temp = w["temperature"].astype(float).values
    feats = pd.DataFrame({
        "block_of_day": block,
        "hour": hour,
        "minute": minute,
        "dayofweek": dow,
        "is_weekend": (dow >= 5).astype(int),
        "month": ts.month,
        "dayofyear": doy,
        "sin_block": np.sin(2 * np.pi * block / BLOCKS_PER_DAY),
        "cos_block": np.cos(2 * np.pi * block / BLOCKS_PER_DAY),
        "sin_doy": np.sin(2 * np.pi * doy / 365.25),
        "cos_doy": np.cos(2 * np.pi * doy / 365.25),
        "temperature": temp,
        "humidity": w["humidity"].astype(float).values,
        "wind_speed": w["wind_speed"].astype(float).values,
        "cloud_cover": w["cloud_cover"].astype(float).values,
        "cooling_degree": np.maximum(0.0, temp - 24.0),
    }, index=ts)
    return pd.concat([feats, holiday_flags(ts, holidays)], axis=1)[FEATURE_COLUMNS]


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_blocks(artifact, feature_frame: pd.DataFrame) -> np.ndarray:
    return artifact["model"].predict(feature_frame[artifact["feature_names"]])


def next_30min_blocks(now: datetime, n_blocks: int = BLOCKS_PER_DAY) -> pd.DatetimeIndex:
    now_ist = now.astimezone(IST)
    floor = now_ist.replace(minute=(0 if now_ist.minute < 30 else 30), second=0, microsecond=0)
    start = floor if now_ist <= floor else floor + pd.Timedelta(minutes=30)
    return pd.date_range(start=start, periods=n_blocks, freq=AGG_FREQ, tz=IST)


def fetch_open_meteo(start_date: str, end_date: str, archive: bool = True,
                     timeout: int = 30) -> pd.DataFrame:
    base = ("https://archive-api.open-meteo.com/v1/archive" if archive
            else "https://api.open-meteo.com/v1/forecast")
    params = {
        "latitude": DHANBAD_LAT, "longitude": DHANBAD_LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m",
        "timezone": "Asia/Kolkata",
    }
    r = requests.get(base, params=params, timeout=timeout)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "ts": pd.to_datetime(h["time"]),
        "temperature": h["temperature_2m"],
        "humidity": h["relative_humidity_2m"],
        "cloud_cover": h["cloud_cover"],
        "wind_speed": h["wind_speed_10m"],
    })
    df["ts"] = df["ts"].dt.tz_localize(IST)
    return df


def hourly_to_30min(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("ts").resample(AGG_FREQ).interpolate("time").round(3).reset_index()
