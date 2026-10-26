"""Wczytuje tabele NHANES z cyklu 2017-2018 używane w tym projekcie i scala je
w jedną ramkę danych per respondent, indeksowaną po SEQN."""

from pathlib import Path

import pandas as pd
import pyreadstat

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "2017-2018"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TABLES = [
    "DEMO_J",
    "BMX_J",
    "BPX_J",
    "GHB_J",
    "GLU_J",
    "DIQ_J",
    "HDL_J",
    "TCHOL_J",
    "TRIGLY_J",
    "INS_J",
    "PAQ_J",
    "SMQ_J",
    "ALQ_J",
]


def load_table(table: str) -> pd.DataFrame:
    path = RAW_DIR / f"{table}.XPT"
    df, _meta = pyreadstat.read_xport(path)
    df["SEQN"] = df["SEQN"].astype(int)
    return df.set_index("SEQN")


def load_merged(tables: list[str] = TABLES) -> pd.DataFrame:
    merged = None
    for table in tables:
        df = load_table(table)
        merged = df if merged is None else merged.join(df, how="outer", suffixes=("", f"_{table}"))
    return merged


def main() -> None:
    merged = load_merged()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "nhanes_2017_2018.parquet"
    merged.to_parquet(out_path)
    print(f"Merged {len(TABLES)} tables -> {merged.shape[0]} respondents, {merged.shape[1]} variables")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
