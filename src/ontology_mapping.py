"""Wyszukuje mapowania ontologiczne (EFO / NCIT / FOODON) dla zmiennych NHANES
oraz bezpośredniego rodzica (is_a) każdego zmapowanego terminu, na podstawie
tabel ccb-hms/NHANES-metadata sklonowanych do external/NHANES-metadata.

Daje to jeden stały poziom grupowania zmiennych "jeden poziom wyżej" w terminy
ontologiczne -- pośredni poziom abstrakcji, od którego projekt startuje, zanim
powstanie adaptacyjna metoda wyboru poziomu kontrolowana pod kątem wierności.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

META_DIR = Path(__file__).resolve().parents[1] / "external" / "NHANES-metadata"
VARIABLE_MAPPINGS = META_DIR / "ontology-mappings" / "nhanes_variables_mappings.tsv"
ONTOLOGY_EDGES = META_DIR / "ontology-tables" / "ontology_edges.tsv"
ONTOLOGY_LABELS = META_DIR / "ontology-tables" / "ontology_labels.tsv"


@dataclass
class OntologyMapping:
    variable: str
    table: str
    source_term: str
    term_label: str
    term_curie: str
    ontology: str
    score: float


def load_variable_mappings(tables: list[str]) -> dict[str, OntologyMapping]:
    """Najlepsze (o najwyższym wyniku) mapowanie ontologiczne dla każdej
    zmiennej NHANES, ograniczone do podanych tabel i do wierszy, które
    faktycznie rozwiązały się do jakiegoś terminu ontologicznego."""
    df = pd.read_csv(VARIABLE_MAPPINGS, sep="\t", dtype=str)
    df["MappingScore"] = pd.to_numeric(df["MappingScore"], errors="coerce")
    df = df[df.Table.isin(tables) & df.MappedTermCURIE.notna()]
    df = df.sort_values("MappingScore", ascending=False).drop_duplicates("Variable")

    mappings: dict[str, OntologyMapping] = {}
    for row in df.itertuples():
        mappings[row.Variable] = OntologyMapping(
            variable=row.Variable,
            table=row.Table,
            source_term=row.SourceTerm,
            term_label=row.MappedTermLabel,
            term_curie=row.MappedTermCURIE,
            ontology=row.Ontology,
            score=row.MappingScore,
        )
    return mappings


def load_parent_labels(curies: list[str]) -> dict[str, str]:
    """Dla każdego CURIE terminu ontologicznego: etykieta jednego z jego
    bezpośrednich rodziców (is_a) (pierwszego znalezionego), używana jako
    klucz grupowania jeden poziom wyżej."""
    curie_set = set(curies)
    edge_rows = []
    with open(ONTOLOGY_EDGES, encoding="utf-8") as fh:
        next(fh)  # nagłówek
        for line in fh:
            subject, obj, _ontology = line.rstrip("\n").split("\t")
            if subject in curie_set:
                edge_rows.append((subject, obj))

    parent_of: dict[str, str] = {}
    for subject, obj in edge_rows:
        parent_of.setdefault(subject, obj)  # zachowaj tylko pierwszego rodzica

    parent_curies = set(parent_of.values())
    label_of: dict[str, str] = {}
    with open(ONTOLOGY_LABELS, encoding="utf-8") as fh:
        next(fh)  # nagłówek
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            subject, label = parts[0], parts[1]
            if subject in parent_curies:
                label_of[subject] = label

    return {
        child: label_of[parent]
        for child, parent in parent_of.items()
        if parent in label_of
    }


def build_feature_groups(feature_vars: list[str], tables: list[str]) -> dict[str, str]:
    """zmienna -> etykieta grupy używana do agregacji wartości SHAP: etykieta
    bezpośredniego rodzica terminu, jeśli jest dostępna, w przeciwnym razie
    własna etykieta terminu, a jeśli nie ma mapowania -- sama nazwa zmiennej."""
    mappings = load_variable_mappings(tables)
    curies = [m.term_curie for v, m in mappings.items() if v in feature_vars]
    parent_labels = load_parent_labels(curies)

    groups: dict[str, str] = {}
    for var in feature_vars:
        mapping = mappings.get(var)
        if mapping is None:
            groups[var] = var
        else:
            groups[var] = parent_labels.get(mapping.term_curie, mapping.term_label)
    return groups


if __name__ == "__main__":
    tables = ["DEMO_J", "BMX_J", "BPX_J", "HDL_J", "TCHOL_J", "SMQ_J", "PAQ_J"]
    feature_vars = [
        "RIAGENDR", "RIDAGEYR", "RIDRETH3", "INDFMPIR", "DMDEDUC2",
        "BMXBMI", "BMXWAIST", "BPXSY1", "BPXDI1",
        "LBDHDD", "LBXTC", "SMQ020", "PAQ605", "PAQ620", "PAQ650", "PAQ665",
    ]
    groups = build_feature_groups(feature_vars, tables)
    for var, group in groups.items():
        print(f"{var:10s} -> {group}")
