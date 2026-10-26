## Projekt w toku

---

# Wyjaśnienia modeli ML wysokiego poziomu oparte na ontologii

Przekształca wyjaśnienia SHAP na poziomie cech (dla modelu wytrenowanego na
danych NHANES) w wyjaśnienia wysokopoziomowe, pogrupowane wg terminów
ontologicznych (EFO/NCIT/FOODON), korzystając z mapowań zmiennych NHANES na
ontologie z repozytorium
[ccb-hms/NHANES-metadata](https://github.com/ccb-hms/NHANES-metadata).

## Instalacja

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./scripts/fetch_nhanes_metadata.sh   # klonuje mapowania ontologiczne do external/
python src/download_nhanes.py        # pobiera dane NHANES 2017-2018 do data/raw/
```

## Uruchomienie

```bash
python src/load_nhanes.py   # scala surowe tabele -> data/processed/nhanes_2017_2018.parquet
python src/train_model.py   # trenuje demonstracyjny klasyfikator cukrzycy, drukuje AUC/metryki
python src/explain.py       # wyjaśnienia SHAP, poziom cech i poziom ontologiczny
```

## Struktura

- `src/load_nhanes.py`, `src/download_nhanes.py` -- pobieranie i scalanie tabel NHANES
- `src/features.py` -- definicje cech/etykiety
- `src/ontology_mapping.py` -- mapowanie zmienna -> termin ontologiczny -> termin nadrzędny
- `src/train_model.py` -- bazowy `HistGradientBoostingClassifier`
- `src/explain.py` -- wyjaśnienia SHAP, na poziomie cech i pogrupowane ontologicznie
- `external/NHANES-metadata` -- sklonowane tabele mapowań/ontologii (zignorowane w git, patrz `scripts/fetch_nhanes_metadata.sh`)
- `data/` -- pobrane/wygenerowane dane (zignorowane w git)
