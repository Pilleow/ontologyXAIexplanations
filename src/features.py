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

FEATURE_TABLES = [
    "DEMO_J", "BMX_J", "BPX_J", "HDL_J", "TCHOL_J", "SMQ_J", "PAQ_J",
    "MCQ_J", "KIQ_U_J", "CDQ_J", "DPQ_J", "BPQ_J", "OSQ_J",
]

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
    "MCQ010",     # czy kiedykolwiek zdiagnozowano astmę
    "MCQ040",     # czy w ostatnich 12 miesiącach był atak astmy
    "KIQ026",     # czy kiedykolwiek miał(a) kamienie nerkowe
    "KIQ029",     # czy w ostatnich 12 miesiącach przeszedł kamień nerkowy
    "MCQ092",     # czy kiedykolwiek miał(a) transfuzję krwi
    "MCQ160B",    # czy kiedykolwiek zdiagnozowano niewydolność serca
    "MCQ160C",    # czy kiedykolwiek zdiagnozowano chorobę wieńcową
    "MCQ160D",    # czy kiedykolwiek zdiagnozowano dusznicę bolesną
    "MCQ160G",    # czy kiedykolwiek zdiagnozowano rozedmę płuc
    "MCQ160K",    # czy kiedykolwiek zdiagnozowano przewlekłe zapalenie oskrzeli
    "MCQ203",     # czy kiedykolwiek miał(a) żółtaczkę
    "CDQ001",     # czy kiedykolwiek miał(a) ból/dyskomfort w klatce piersiowej
    "CDQ010",     # czy ma duszność przy wchodzeniu pod górę/szybkim chodzeniu
    "DPQ010",     # PHQ-9: mało zainteresowania/przyjemności z robienia rzeczy
    "DPQ020",     # PHQ-9: poczucie smutku/depresji/bezradności
    "BPQ020",     # czy kiedykolwiek zdiagnozowano nadciśnienie
    "BPQ080",     # czy kiedykolwiek zdiagnozowano wysoki cholesterol
    "OSQ060",     # czy kiedykolwiek zdiagnozowano osteoporozę
    "MCQ160E",    # czy kiedykolwiek miał(a) zawał serca
    "MCQ160N",    # czy kiedykolwiek zdiagnozowano dnę moczanową
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
    "MCQ010": [7, 9],
    "MCQ040": [7, 9],
    "KIQ026": [7, 9],
    "KIQ029": [7, 9],
    "MCQ092": [7, 9],
    "MCQ160B": [7, 9],
    "MCQ160C": [7, 9],
    "MCQ160D": [7, 9],
    "MCQ160G": [7, 9],
    "MCQ160K": [7, 9],
    "MCQ203": [7, 9],
    "CDQ001": [7, 9],
    "CDQ010": [7, 9],
    "DPQ010": [7, 9],
    "DPQ020": [7, 9],
    "BPQ020": [7, 9],
    "BPQ080": [7, 9],
    "OSQ060": [7, 9],
    "MCQ160E": [7, 9],
    "MCQ160N": [7, 9],
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
