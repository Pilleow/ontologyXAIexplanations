"""Wyszukuje mapowania ontologiczne (EFO / NCIT / FOODON) dla zmiennych NHANES
oraz przodków (is_a) każdego zmapowanego terminu, na podstawie tabel
ccb-hms/NHANES-metadata sklonowanych do external/NHANES-metadata.

Grupowanie "jeden poziom wyżej" jest sterowane progiem na skali
specificity_score (`target_specificity`, ta sama skala co w
term_specificity.py: 1.0 = liść, 0.0 = korzeń), nie ustaloną liczbą hopów --
dzięki temu interfejs (build_feature_groups) nie zmieni się, gdy
specificity_score zostanie kiedyś wzbogacony o CHV/UMLS, wyszukiwania czy
embeddingi (patrz plan dot. "eksperckości" terminu), zamiast samej głębokości.
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


def load_labels(curies: set[str]) -> dict[str, str]:
    """Etykieta (label) dla podanego zbioru CURIE, z ontology_labels.tsv."""
    label_of: dict[str, str] = {}
    with open(ONTOLOGY_LABELS, encoding="utf-8") as fh:
        next(fh)  # nagłówek
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            subject, label = parts[0], parts[1]
            if subject in curies:
                label_of[subject] = label
    return label_of


def load_first_parent_map(curies: set[str]) -> dict[str, str]:
    """Dla podanego zbioru CURIE (jedna "warstwa" wspinaczki): jeden,
    pierwszy znaleziony w pliku bezpośredni rodzic (is_a) każdego z nich.
    Wybór "pierwszego" jest zdeterminowany kolejnością wierszy w
    ontology_edges.tsv (ten sam mechanizm, co dotychczasowy kod)."""
    parent_of: dict[str, str] = {}
    with open(ONTOLOGY_EDGES, encoding="utf-8") as fh:
        next(fh)  # nagłówek
        for line in fh:
            subject, obj, _ontology = line.rstrip("\n").split("\t")
            if subject in curies and subject not in parent_of:
                parent_of[subject] = obj
    return parent_of


def ancestor_chains(leaf_curies: set[str]) -> dict[str, list[str]]:
    """Dla każdego liścia: pełny, zdeterminowany łańcuch (pierwszy znaleziony
    rodzic na każdym kroku -- ten sam mechanizm co dotychczasowy kod) od
    liścia (chain[0]) do korzenia (chain[-1], węzła bez dalszego rodzica),
    włącznie z obydwoma końcami."""
    chains: dict[str, list[str]] = {c: [c] for c in leaf_curies}
    current: dict[str, str] = {c: c for c in leaf_curies}
    active = set(leaf_curies)
    while active:
        parent_of = load_first_parent_map({current[c] for c in active})
        still_active = set()
        for c in active:
            parent = parent_of.get(current[c])
            if parent is not None:
                chains[c].append(parent)
                current[c] = parent
                still_active.add(c)
        active = still_active
    return chains


def specificity_along_chain(chain: list[str]) -> list[float]:
    """specificity_score dla każdej pozycji w `chain` (ta sama formuła co w
    term_specificity.specificity_scores, tylko ograniczona do tego jednego
    zdeterminowanego łańcucha): chain[0] (liść) -> 1.0, chain[-1] (korzeń
    osiągalny z tego liścia) -> 0.0."""
    max_dist = len(chain) - 1
    if max_dist == 0:
        return [1.0]
    return [1.0 - i / max_dist for i in range(len(chain))]


def climb_to_specificity(leaf_curies: set[str], target_specificity: float) -> dict[str, str]:
    """Dla każdego liścia: CURIE osiągnięty po wspinaczce po jego
    zdeterminowanym łańcuchu "pierwszego rodzica" aż specificity_score
    spadnie do/poniżej `target_specificity` (1.0 -> zostaje na liściu,
    0.0 -> idzie do korzenia)."""
    chains = ancestor_chains(leaf_curies)
    reached: dict[str, str] = {}
    for leaf, chain in chains.items():
        scores = specificity_along_chain(chain)
        reached[leaf] = next(
            (curie for curie, score in zip(chain, scores) if score <= target_specificity),
            chain[-1],
        )
    return reached


def load_parent_labels(curies: list[str]) -> dict[str, str]:
    """Dla każdego CURIE terminu ontologicznego: etykieta jego bezpośredniego
    rodzica (is_a, pierwszego znalezionego), używana jako klucz grupowania
    jeden poziom wyżej."""
    chains = ancestor_chains(set(curies))
    reached = {c: chain[1] for c, chain in chains.items() if len(chain) > 1}
    label_of = load_labels(set(reached.values()))
    return {child: label_of[parent] for child, parent in reached.items() if parent in label_of}


def build_feature_groups(
    feature_vars: list[str], tables: list[str], target_specificity: float = 1.0
) -> dict[str, str]:
    """zmienna -> etykieta grupy używana do agregacji wartości SHAP: etykieta
    pierwszego przodka na ścieżce do korzenia, którego specificity_score
    spadł do/poniżej target_specificity (target_specificity=1.0: etykieta
    samego terminu/liścia, bez grupowania; 0.0: korzeń); jeśli osiągnięty
    węzeł nie ma własnej etykiety w ontology_labels.tsv (np. owl:Thing,
    IAO:0000030), użyty jest jego CURIE -- NIE etykieta liścia, żeby nie
    zamaskować tego, że wspinaczka faktycznie dotarła gdzie indziej; a jeśli
    nie ma mapowania -- sama nazwa zmiennej."""
    mappings = load_variable_mappings(tables)
    curies = {m.term_curie for v, m in mappings.items() if v in feature_vars}
    reached = climb_to_specificity(curies, target_specificity)
    label_of = load_labels(set(reached.values()))

    groups: dict[str, str] = {}
    for var in feature_vars:
        mapping = mappings.get(var)
        if mapping is None:
            groups[var] = var
        else:
            reached_curie = reached.get(mapping.term_curie, mapping.term_curie)
            groups[var] = label_of.get(reached_curie, reached_curie)
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
