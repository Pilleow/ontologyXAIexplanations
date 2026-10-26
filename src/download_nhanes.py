"""Pobiera tabele NHANES z cyklu 2017-2018 używane w tym projekcie z serwera
danych continuous NHANES (CDC)."""

from pathlib import Path

import requests

from load_nhanes import TABLES, RAW_DIR

CYCLE_BEGIN_YEAR = 2017
BASE_URL = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{CYCLE_BEGIN_YEAR}/DataFiles"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        out_path = RAW_DIR / f"{table}.XPT"
        if out_path.exists():
            print(f"skip {table} (already downloaded)")
            continue
        url = f"{BASE_URL}/{table}.xpt"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        out_path.write_bytes(response.content)
        print(f"downloaded {table} ({len(response.content)} bytes)")


if __name__ == "__main__":
    main()
