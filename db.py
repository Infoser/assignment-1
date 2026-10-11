import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
TIMEOUT_SECONDS = 10


def _load_dotenv():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


def supabase_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)


def _headers():
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }


def sb_select(table: str, params: dict | None = None):
    if not supabase_enabled():
        return None
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_headers(),
            params=params or {},
            timeout=TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def sb_upsert(table: str, rows: list) -> bool:
    if not supabase_enabled():
        return False
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers={**_headers(), "Prefer": "resolution=merge-duplicates"},
            json=rows,
            timeout=TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        return True
    except Exception:
        return False


def sb_insert(table: str, rows: list) -> bool:
    if not supabase_enabled():
        return False
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_headers(),
            json=rows,
            timeout=TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        return True
    except Exception:
        return False


def supabase_reachable() -> bool:
    return sb_select("holidays", {"select": "id", "limit": 1}) is not None
