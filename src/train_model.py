"""Trenuje bazowy klasyfikator typu gradient-boosted trees dla cukrzycy
(HbA1c >= 6.5%) na cechach NHANES 2017-2018. To model demonstracyjny dla
pipeline'u wyjaśnień, a nie dostrojony predyktor kliniczny.

Używa HistGradientBoostingClassifier ze sklearn (natywnie obsługuje braki
danych, nie wymaga zewnętrznego runtime'u OpenMP) zamiast XGBoost."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from features import build_dataset

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_PATH = PROCESSED_DIR / "diabetes_model.joblib"


def main() -> None:
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    model = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=200,
        random_state=0,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, pred, digits=3))

    joblib.dump(model, MODEL_PATH)
    X_test.to_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test.to_frame("label").to_parquet(PROCESSED_DIR / "y_test.parquet")
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
