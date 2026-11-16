"""Buduje macierz cech X i etykietę cukrzycy y na podstawie scalonej ramki
danych NHANES 2017-2018.

Etykieta: HbA1c (LBXGH) >= 6.5%, standardowy próg diagnostyczny ADA dla
cukrzycy -- obiektywne kryterium laboratoryjne, dlatego same zmienne
diagnostyczne (LBXGH oraz kwestionariusz samoopisowy DIQ_J) są wykluczone ze
zbioru cech, żeby uniknąć przecieku etykiety do predyktorów.
"""

import pandas as pd

from config.feature_selection import (
    DIABETES_THRESHOLD,
    FEATURE_TABLES,
    FEATURE_VARS,
    LABEL_VAR,
    MISSING_CODES,
)
from config.paths import PROCESSED_DIR


def build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DIR / "nhanes_2017_2018.parquet")

    for var, codes in MISSING_CODES.items():
        df.loc[df[var].isin(codes), var] = pd.NA

    labelled = df[df[LABEL_VAR].notna()].copy()
    y = (labelled[LABEL_VAR] >= DIABETES_THRESHOLD).astype(int)
    X = labelled[FEATURE_VARS].copy()

    return X, y


if __name__ == "__main__":
    X, y = build_dataset()
    print(f"Dataset: {X.shape[0]} respondents, {X.shape[1]} features")
    print(f"Diabetes prevalence (HbA1c >= {DIABETES_THRESHOLD}%): {y.mean():.1%}")
    print(X.isna().mean().sort_values(ascending=False).rename("missing_frac"))
