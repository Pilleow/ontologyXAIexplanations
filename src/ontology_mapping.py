"""Wyszukuje mapowania ontologiczne (EFO / NCIT / FOODON) dla zmiennych NHANES
oraz przodków (is_a) każdego zmapowanego terminu, na podstawie tabel
ccb-hms/NHANES-metadata sklonowanych do external/NHANES-metadata.

Grupowanie "jeden poziom wyżej" jest sterowane progiem na skali
specificity_score (`target_specificity`, ta sama skala co w
term_specificity.py: 1.0 = liść, 0.0 = korzeń), nie ustaloną liczbą hopów.
specificity_score jest, gdzie to możliwe, oparty na dopasowaniu etykiety do
Consumer Health Vocabulary (chv_familiarity.py); dla węzłów bez trafienia w
CHV zostaje dotychczasowy, czysto strukturalny fallback z głębokości.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chv_familiarity import chv_familiarity, load_chv_scores

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


# Poprawki tam, gdzie oryginalne mapowanie (upstream ccb-hms/NHANES-metadata,
# text2term na całym tekście pytania ankietowego) albo nie wygenerowało
# żadnego mapowania, albo zgubiło kluczowe słowo z pytania. Każda poprawka
# jest zweryfikowana ponownym przebiegiem text2term na skróconej, poprawnej
# frazie (zamiast całego szablonu pytania) względem naszego własnego,
# lokalnego zbioru terminów EFO/NCIT/FOODON -- patrz src/fix_mappings.py,
# nie czysto ręczne asercje CURIE.
MANUAL_OVERRIDES: dict[str, tuple[str, str]] = {
    # "chest pain" -> Chest Pain (wynik 0.982). Oryginalnie dopasowane do
    # ogólnego NCIT:C156140 "Pain or Discomfort" (wynik 0.743), gubiąc słowo
    # "chest". CDQ001 jest #4 wg globalnego |SHAP| w tym modelu.
    "CDQ001": ("NCIT:C38665", "Chest Pain"),
    # "high cholesterol" -> najlepszy kandydat to ogólne "Cholesterol"
    # (NCIT:C369, 0.842), ale "Cholesterol High, CTCAE" (NCIT:C143368,
    # 0.742) lepiej odpowiada treści pytania (podwyższony, nie sam pomiar)
    # -- wybór semantyczny, nie czysto najwyższy wynik. Oryginalnie brak
    # mapowania w ogóle (score 0.0) dla tego cyklu.
    "BPQ080": ("NCIT:C143368", "Cholesterol High, CTCAE"),
    # "heart attack" nie ma dobrego kandydata w text2term (najlepszy:
    # "Heart" 0.673, zupełnie ogólny). Używamy więc CURIE, które upstream
    # poprawnie przypisał siostrzanej zmiennej o wieku rozpoznania tego
    # samego schorzenia (MCD180E), zamiast zgadywać od nowa.
    "MCQ160E": ("NCIT:C27996", "Myocardial Infarction"),
    # "gout" -> gout/Gout (wynik 0.996, prawie idealny). Oryginalnie brak
    # mapowania w ogóle dla tej (nie wiekowej) zmiennej.
    "MCQ160N": ("EFO:0004274", "gout"),
}


def load_variable_mappings(tables: list[str]) -> dict[str, OntologyMapping]:
    """Najlepsze (o najwyższym wyniku) mapowanie ontologiczne dla każdej
    zmiennej NHANES, ograniczone do podanych tabel i do wierszy, które
    faktycznie rozwiązały się do jakiegoś terminu ontologicznego, z
    naniesionymi MANUAL_OVERRIDES."""
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

    for var, (curie, label) in MANUAL_OVERRIDES.items():
        existing = mappings.get(var)
        mappings[var] = OntologyMapping(
            variable=var,
            table=existing.table if existing else "",
            source_term=existing.source_term if existing else "",
            term_label=label,
            term_curie=curie,
            ontology=curie.split(":")[0],
            score=1.0,
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


def specificity_along_chain_with_chv(
    chain: list[str], label_of: dict[str, str], chv_scores: dict[str, float]
) -> list[float]:
    """Jak specificity_along_chain, ale dla każdego węzła, którego etykieta
    trafia w CHV (patrz chv_familiarity.py), podstawia 1 - familiarity
    zamiast czysto strukturalnej wartości -- również dla samego liścia
    (jeśli liść jest już powszechnie znanym słowem, nie trzeba się wspinać
    wyżej, mimo że strukturalnie liść ma zawsze specificity=1.0). Dla
    węzłów bez trafienia w CHV, które leżą MIĘDZY dwoma węzłami z
    trafieniem, wartość jest interpolowana liniowo między nimi (dla
    jednego węzła pośrodku to po prostu średnia arytmetyczna sąsiadów) --
    lepiej oddaje lokalne sąsiedztwo niż niezwiązana wartość strukturalna.
    Tam, gdzie nie ma kotwicy CHV po jednej ze stron (np. ogon łańcucha
    w stronę owl:Thing), zostaje dotychczasowy fallback strukturalny.
    Wynikowa sekwencja nie musi być monotoniczna -- to świadome: realna
    znajomość słowa nie rośnie liniowo z głębokością w ontologii (patrz
    plan, sekcja 5)."""
    structural = specificity_along_chain(chain)
    chv_values: list[float | None] = []
    for curie in chain:
        label = label_of.get(curie)
        fam = chv_familiarity(label, chv_scores) if label else None
        chv_values.append(1.0 - fam if fam is not None else None)

    anchors = [i for i, v in enumerate(chv_values) if v is not None]
    scores = list(chv_values)
    for i, value in enumerate(chv_values):
        if value is not None:
            continue
        before = max((a for a in anchors if a < i), default=None)
        after = min((a for a in anchors if a > i), default=None)
        if before is not None and after is not None:
            frac = (i - before) / (after - before)
            scores[i] = chv_values[before] + frac * (chv_values[after] - chv_values[before])
        else:
            scores[i] = structural[i]
    return scores


def climb_to_specificity(leaf_curies: set[str], target_specificity: float) -> dict[str, str]:
    """Dla każdego liścia: CURIE osiągnięty po wspinaczce po jego
    zdeterminowanym łańcuchu "pierwszego rodzica" aż specificity_score
    (najpierw z CHV przez dopasowanie tekstowe etykiety, w przeciwnym razie
    strukturalny) spadnie do/poniżej `target_specificity` (1.0 -> zostaje
    na liściu, 0.0 -> idzie do korzenia)."""
    chains = ancestor_chains(leaf_curies)
    label_of = load_labels({c for chain in chains.values() for c in chain})
    chv_scores = load_chv_scores()

    reached: dict[str, str] = {}
    for leaf, chain in chains.items():
        scores = specificity_along_chain_with_chv(chain, label_of, chv_scores)
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
