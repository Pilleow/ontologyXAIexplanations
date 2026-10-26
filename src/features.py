"""Buduje macierz cech X i etykietę cukrzycy y na podstawie scalonej ramki
danych NHANES 2017-2018.

Etykieta: HbA1c (LBXGH) >= 6.5%, standardowy próg diagnostyczny ADA dla
cukrzycy -- obiektywne kryterium laboratoryjne, dlatego same zmienne
diagnostyczne (LBXGH oraz kwestionariusz samoopisowy DIQ_J) są wykluczone ze
zbioru cech, żeby uniknąć przecieku etykiety do predyktorów.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

FEATURE_TABLES = ["DEMO_J", "BMX_J", "BPX_J", "HDL_J", "TCHOL_J", "SMQ_J", "PAQ_J"]

FEATURE_VARS = [
    "RIAGENDR",   # płeć
    "RIDAGEYR",   # wiek w latach
    "RIDRETH3",   # rasa/pochodzenie etniczne
    "INDFMPIR",   # stosunek dochodu rodziny do progu ubóstwa
    "DMDEDUC2",   # poziom wykształcenia, dorośli
    "BMXBMI",     # wskaźnik masy ciała (BMI)
    "BMXWAIST",   # obwód talii
    "BPXSY1",     # ciśnienie skurczowe
    "BPXDI1",     # ciśnienie rozkurczowe
    "LBDHDD",     # cholesterol HDL
    "LBXTC",      # cholesterol całkowity
    "SMQ020",     # czy kiedykolwiek wypalił(a) 100 papierosów
    "PAQ605",     # intensywna aktywność w pracy
    "PAQ620",     # umiarkowana aktywność w pracy
    "PAQ650",     # intensywna aktywność rekreacyjna
    "PAQ665",     # umiarkowana aktywność rekreacyjna
]

LABEL_VAR = "LBXGH"
DIABETES_THRESHOLD = 6.5

# Kodebooki NHANES używają 7/9 (lub 77/99) dla "Odmowa"/"Nie wiem" w pytaniach
# kategorycznych kwestionariusza; traktujemy je jako braki danych, a nie jako
# prawidłowe kategorie.
MISSING_CODES = {
    "DMDEDUC2": [7, 9],
    "SMQ020": [7, 9],
    "PAQ605": [7, 9],
    "PAQ620": [7, 9],
    "PAQ650": [7, 9],
    "PAQ665": [7, 9],
}


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
