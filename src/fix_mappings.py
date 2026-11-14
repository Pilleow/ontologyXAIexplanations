"""Ponownie puszcza text2term (ten sam mapper TF-IDF co upstream
ccb-hms/NHANES-metadata) na POPRAWIONYCH frazach wejściowych dla zmiennych,
gdzie oryginalne mapowanie dopasowało szablon pytania ankietowego, nie
symptom (patrz MANUAL_OVERRIDES w ontology_mapping.py) -- zamiast ręcznie
asertować CURIE, dostajemy realny, zweryfikowany wynik dopasowania.

Mapuje lokalnie, na własnym, już sklonowanym ontology_labels.tsv (246k
terminów EFO/NCIT/FOODON) -- bez potrzeby pobierania pełnych plików OWL czy
klucza API BioPortal.
"""

from pathlib import Path

from text2term.term import OntologyTerm
from text2term.tfidf_mapper import TFIDFMapper

META_DIR = Path(__file__).resolve().parent.parent / "external" / "NHANES-metadata"
ONTOLOGY_LABELS = META_DIR / "ontology-tables" / "ontology_labels.tsv"

# Frazy poprawione względem oryginalnego, pełnego pytania ankietowego --
# sam symptom/koncept, nie szablon pytania (patrz plan: "naprawa mapowania
# u źródła" jako alternatywa do ręcznych MANUAL_OVERRIDES).
CORRECTED_SOURCE_TERMS = {
    "CDQ001": "chest pain",
    "BPQ080": "high cholesterol",
    "MCQ160E": "heart attack",
    "MCQ160N": "gout",
}


def load_target_terms() -> dict[str, OntologyTerm]:
    """Wszystkie terminy EFO/NCIT/FOODON jako OntologyTerm (CURIE -> term),
    z naszego już sklonowanego ontology_labels.tsv."""
    terms: dict[str, OntologyTerm] = {}
    with open(ONTOLOGY_LABELS, encoding="utf-8") as fh:
        next(fh)  # nagłówek
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            curie, label = parts[0], parts[1]
            terms[curie] = OntologyTerm(iri=curie, labels={label})
    return terms


def remap_with_text2term(source_terms: dict[str, str]) -> None:
    """Mapuje poprawione frazy i wypisuje najlepsze dopasowania z realnym
    wynikiem ufności."""
    target_terms = load_target_terms()
    mapper = TFIDFMapper(target_terms)
    results = mapper.map(
        list(source_terms.values()),
        source_terms_ids=list(source_terms.keys()),
        max_mappings=3,
        min_score=0.5,
    )
    print(results.columns.tolist())
    print(results.to_string())


if __name__ == "__main__":
    remap_with_text2term(CORRECTED_SOURCE_TERMS)
