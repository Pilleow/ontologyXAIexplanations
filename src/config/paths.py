# Ścieżki do danych NHANES, metadanych ontologii i pliku CHV.

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = REPO_ROOT / "data" / "raw" / "2017-2018"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODEL_PATH = PROCESSED_DIR / "diabetes_model.joblib"

META_DIR = REPO_ROOT / "external" / "NHANES-metadata"
VARIABLE_MAPPINGS = META_DIR / "ontology-mappings" / "nhanes_variables_mappings.tsv"
ONTOLOGY_LABELS = META_DIR / "ontology-tables" / "ontology_labels.tsv"
ONTOLOGY_EDGES = META_DIR / "ontology-tables" / "ontology_edges.tsv"

CHV_FILE = REPO_ROOT / "data" / "raw" / "CHV" / "CHV_concepts_terms_flatfile_20110204.tsv"
