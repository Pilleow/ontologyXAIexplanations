"""Generuje wyjaśnienia SHAP na poziomie cech dla bazowego modelu cukrzycy,
a następnie agreguje je do wyjaśnień wysokopoziomowych opartych na ontologii,
używając mapowania zmienna -> grupa ontologiczna z ontology_mapping.py.

To pierwsza wersja metody wyjaśniania o ustalonym, stałym poziomie
abstrakcji: każda zmapowana cecha jest podciągana dokładnie jeden poziom
wyżej (do bezpośredniego rodzica w ontologii), a jej wkład SHAP jest sumowany
w tej grupie. Adaptacyjny wybór poziomu abstrakcji i pomiar wynikającej z
tego utraty wierności/informacji to wciąż otwarty problem metodologiczny,
który projekt musi rozwiązać.
"""

from pathlib import Path

import joblib
import pandas as pd
import shap

from features import FEATURE_TABLES, FEATURE_VARS
from ontology_mapping import build_feature_groups

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def grouped_shap(shap_row: pd.Series, groups: dict[str, str]) -> pd.Series:
    labelled = shap_row.rename(index=lambda v: groups.get(v, v))
    return labelled.groupby(level=0).sum().sort_values(key=abs, ascending=False)


def main(n_examples: int = 3) -> None:
    model = joblib.load(PROCESSED_DIR / "diabetes_model.joblib")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["label"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    shap_df = pd.DataFrame(shap_values.values, columns=FEATURE_VARS, index=X_test.index)

    groups = build_feature_groups(FEATURE_VARS, FEATURE_TABLES)
    print("Feature -> ontology group used for aggregation:")
    for var, group in groups.items():
        print(f"  {var:10s} -> {group}")

    proba = model.predict_proba(X_test)[:, 1]
    # wybierz kilku pacjentów wysokiego ryzyka (prawdziwie pozytywnych) jako przykłady ilustracyjne
    candidates = X_test.index[(y_test == 1) & (proba >= 0.5)]
    examples = list(candidates[:n_examples]) or list(X_test.index[:n_examples])

    for seqn in examples:
        print(f"\n=== Respondent SEQN={seqn}  P(diabetes)={proba[X_test.index.get_loc(seqn)]:.2f} ===")

        print("-- Feature-level SHAP (top 6) --")
        feat_shap = shap_df.loc[seqn].sort_values(key=abs, ascending=False)
        for var, val in feat_shap.head(6).items():
            print(f"  {var:10s} {val:+.3f}  (value={X_test.loc[seqn, var]})")

        print("-- Ontology-grouped SHAP (top 6) --")
        group_shap = grouped_shap(shap_df.loc[seqn], groups)
        for group, val in group_shap.head(6).items():
            print(f"  {group:35s} {val:+.3f}")


if __name__ == "__main__":
    main()
