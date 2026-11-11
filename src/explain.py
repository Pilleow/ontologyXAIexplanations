"""Generuje wyjaśnienia SHAP na poziomie cech dla bazowego modelu cukrzycy,
a następnie agreguje je do wyjaśnień wysokopoziomowych opartych na ontologii,
używając mapowania zmienna -> grupa ontologiczna z ontology_mapping.py.

Poziom abstrakcji jest na razie ustalany ręcznie, jednym progiem wspólnym dla
wszystkich cech (parametr `--target-specificity`, skala specificity_score z
term_specificity.py: 1.0 = liść/bez grupowania, 0.0 = korzeń): każda
zmapowana cecha jest podciągana po swojej ścieżce do korzenia, aż jej
specificity_score spadnie do/poniżej tego progu, a jej wkład SHAP jest
sumowany w tak wybranej grupie. Adaptacyjny, zróżnicowany próg per cecha --
np. gdy specificity_score zostanie wzbogacony o CHV/UMLS lub wyszukiwania,
nie tylko głębokość -- i pomiar wynikającej z tego utraty wierności/
informacji to wciąż otwarty problem metodologiczny, który projekt musi
rozwiązać.
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


def main(n_examples: int = 3, target_specificity: float = 1.0) -> None:
    model = joblib.load(PROCESSED_DIR / "diabetes_model.joblib")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["label"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    shap_df = pd.DataFrame(shap_values.values, columns=FEATURE_VARS, index=X_test.index)

    groups = build_feature_groups(
        FEATURE_VARS, FEATURE_TABLES, target_specificity=target_specificity
    )
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Wyjaśnienia SHAP na poziomie cech i pogrupowane ontologicznie."
    )
    parser.add_argument(
        "--target-specificity",
        type=float,
        default=1.0,
        help=(
            "Próg na skali specificity_score (1.0 = liść/bez grupowania "
            "[domyślnie], 0.0 = korzeń), do którego wspinamy się po ontologii "
            "przed zsumowaniem SHAP."
        ),
    )
    parser.add_argument(
        "--n-examples", type=int, default=3, help="Liczba przykładowych respondentów."
    )
    args = parser.parse_args()
    main(n_examples=args.n_examples, target_specificity=args.target_specificity)
