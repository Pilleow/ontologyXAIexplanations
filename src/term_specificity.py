"""Fundament scoringu "eksperckości" (specyficzności) terminu ontologicznego.

Dla każdego zmapowanego liścia (terminu EFO/NCIT, do którego text2term dopasował
zmienną NHANES) liczy pełną mapę przodków (is_a, poprzez ontology_edges.tsv --
WSZYSTKICH rodziców danego węzła, w przeciwieństwie do ontology_mapping.load_parent_labels,
które zachowuje tylko jednego, pierwszego znalezionego) i głębokość (liczbę hopów)
każdego przodka względem tego liścia.

specificity_score węzła jest znormalizowany LOKALNIE względem konkretnego liścia:
0 = korzeń osiągalny z tego liścia (węzeł bez dalszych rodziców), 1 = sam liść.
Normalizacja jest świadomie lokalna, nie globalna -- głębokość w DAGu EFO/NCIT nie
jest porównywalna między różnymi gałęziami (różne gałęzie są kuratorowane z różną
szczegółowością), więc lokalna normalizacja przynajmniej gwarantuje pełny zakres 0-1
dla każdej konkretnej ścieżki wyjaśnienia. Patrz plan: podejście 4 (baza/fallback,
zero zależności zewnętrznych) -- docelowo ma być zastępowane/uzupełniane przez
podejścia 1-3 tam, gdzie jest dla nich pokrycie danych.

"Korzeń" nie jest tu hardkodowany (nie zakładamy konkretnego CURIE) -- jest wykrywany
z danych: to każdy węzeł osiągalny z liścia, który nie ma już własnych rodziców w
ontology_edges.tsv.
"""

from pathlib import Path

ONTOLOGY_EDGES = (
    Path(__file__).resolve().parent.parent
    / "external"
    / "NHANES-metadata"
    / "ontology-tables"
    / "ontology_edges.tsv"
)


def load_parent_map() -> dict[str, set[str]]:
    """child CURIE -> zbiór WSZYSTKICH bezpośrednich rodziców (is_a)."""
    parent_map: dict[str, set[str]] = {}
    with open(ONTOLOGY_EDGES, encoding="utf-8") as fh:
        next(fh)  # nagłówek
        for line in fh:
            subject, obj, _ontology = line.rstrip("\n").split("\t")
            parent_map.setdefault(subject, set()).add(obj)
    return parent_map


def specificity_scores(leaf_curie: str, parent_map: dict[str, set[str]]) -> dict[str, float]:
    """specificity_score każdego przodka leaf_curie (i samego leaf_curie),
    znormalizowany lokalnie względem najdłuższej ścieżki tego liścia do
    osiągalnego korzenia: 0 = korzeń, 1 = sam liść."""
    dist_from_leaf: dict[str, int] = {leaf_curie: 0}
    frontier = [leaf_curie]
    while frontier:
        next_frontier = []
        for node in frontier:
            for parent in parent_map.get(node, ()):
                if parent not in dist_from_leaf:
                    dist_from_leaf[parent] = dist_from_leaf[node] + 1
                    next_frontier.append(parent)
        frontier = next_frontier

    max_dist = max(dist_from_leaf.values())
    if max_dist == 0:
        return {leaf_curie: 1.0}
    return {node: 1.0 - dist / max_dist for node, dist in dist_from_leaf.items()}


if __name__ == "__main__":
    from features import FEATURE_TABLES, FEATURE_VARS
    from ontology_mapping import load_labels, load_variable_mappings

    mappings = load_variable_mappings(FEATURE_TABLES)
    parent_map = load_parent_map()

    for var in FEATURE_VARS:
        mapping = mappings.get(var)
        if mapping is None:
            print(f"\n{var} -> (brak mapowania ontologicznego)")
            continue

        scores = specificity_scores(mapping.term_curie, parent_map)
        labels = load_labels(set(scores) - {mapping.term_curie})
        labels[mapping.term_curie] = mapping.term_label

        print(f"\n{var} (liść: {mapping.term_label}, {mapping.term_curie}):")
        for curie, score in sorted(scores.items(), key=lambda kv: kv[1]):
            label = labels.get(curie, curie)
            print(f"  specificity={score:.3f}  {label}  ({curie})")
