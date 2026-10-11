import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from db import supabase_enabled, sb_upsert


def main():
    if not supabase_enabled():
        print("SUPABASE_URL / SUPABASE_SERVICE_KEY missing - set them in .env or environment")
        sys.exit(1)

    data = json.loads((ROOT / "data" / "holidays_dhanbad.json").read_text(encoding="utf-8"))
    rows = [
        {k: h.get(k, "") for k in ("date", "name", "category", "impact", "notes")}
        for h in data["holidays"]
    ]
    ok = sb_upsert("holidays", rows)
    if ok:
        print(f"seeded/updated {len(rows)} holiday rows in the Supabase 'holidays' table")
    else:
        print("upsert failed - check that supabase/schema.sql was run and the service key is valid")
        sys.exit(1)


if __name__ == "__main__":
    main()
