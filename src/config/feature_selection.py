# Które zmienne NHANES wchodzą do modelu jako cechy, etykieta cukrzycy i braki danych.

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
    "MCQ010",     # astma kiedykolwiek
    "MCQ040",     # atak astmy w ostatnich 12 mies.
    "KIQ026",     # kamienie nerkowe kiedykolwiek
    "KIQ029",     # kamień nerkowy w ostatnich 12 mies.
    "MCQ092",     # transfuzja krwi kiedykolwiek
    "MCQ160B",    # niewydolność serca
    "MCQ160C",    # choroba wieńcowa
    "MCQ160D",    # dusznica bolesna
    "MCQ160G",    # rozedma płuc
    "MCQ160K",    # przewlekłe zapalenie oskrzeli
    "MCQ203",     # żółtaczka
    "CDQ001",     # ból/dyskomfort w klatce piersiowej
    "CDQ010",     # duszność przy wchodzeniu pod górę/szybkim chodzeniu
    "DPQ010",     # PHQ-9: mało zainteresowania/przyjemności
    "DPQ020",     # PHQ-9: smutek/depresja/bezradność
    "BPQ020",     # nadciśnienie
    "BPQ080",     # wysoki cholesterol
    "OSQ060",     # osteoporoza
    "MCQ160E",    # zawał serca
    "MCQ160N",    # dna moczanowa
]

LABEL_VAR = "LBXGH"
DIABETES_THRESHOLD = 6.5  # HbA1c >= 6.5%, próg ADA

# NHANES: 7/9 (lub 77/99) = "Odmowa"/"Nie wiem" -> traktowane jako braki danych.
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
