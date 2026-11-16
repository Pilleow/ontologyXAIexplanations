"""Wczytuje tabele NHANES z cyklu 2017-2018 używane w tym projekcie i scala je
w jedną ramkę danych per respondent, indeksowaną po SEQN."""

import pandas as pd
import pyreadstat

from config.nhanes_tables import TABLES
from config.paths import PROCESSED_DIR, RAW_DIR


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
