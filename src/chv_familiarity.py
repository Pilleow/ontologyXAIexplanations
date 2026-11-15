"""Dopasowuje etykiety terminów ontologii do pliku Consumer Health Vocabulary
(CHV) przez sam tekst etykiety -- bez pośrednictwa UMLS CUI/dbxrefs, które
pokrywają tylko niewielką część terminów EFO/NCIT (patrz plan dot.
"eksperckości" terminu, podejście 2).

Plik CHV (data/raw/CHV/CHV_concepts_terms_flatfile_20110204.tsv) jest bez
nagłówka. Kolumna 2 (indeks 1) to wariant słowny (string); kolumna 9
(indeks 8) to familiarity score (0=niezrozumiałe dla laika, 1=powszechnie
znane); kolumna 6 (indeks 5) to flaga "czy to preferowany wariant CHV dla
danego konceptu" -- sprawdzone empirycznie na danych, nie z oficjalnej
dokumentacji (strona z ReadMe miała problem z certyfikatem SSL).
"""

from pathlib import Path

CHV_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "CHV"
    / "CHV_concepts_terms_flatfile_20110204.tsv"
)


def load_chv_scores() -> dict[str, float]:
    """wariant słowny (małe litery) -> familiarity score. Gdy ten sam
    string występuje dla więcej niż jednego konceptu UMLS, zachowuje
    wariant oznaczony jako preferowany przez CHV."""
    scores: dict[str, float] = {}
    preferred: set[str] = set()
    with open(CHV_FILE, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            string, is_preferred, score = parts[1], parts[5], parts[8]
            if string in preferred:
                continue
            scores[string] = float(score)
            if is_preferred == "yes":
                preferred.add(string)
    return scores


def label_variants(label: str) -> list[str]:
    """Kandydujące uproszczenia etykiety ontologii do spróbowania w CHV,
    od najbardziej dosłownego do najbardziej uproszczonego (np. obcięcie
    sufiksu ", CTCAE" / ", CTCAE 5.0" używanego przez NCIT)."""
    base = label.lower().strip()
    variants = [base]
    if "," in base:
        variants.append(base.split(",")[0].strip())
    return variants


def chv_familiarity(label: str, chv_scores: dict[str, float]) -> float | None:
    """familiarity score (0-1) dla pierwszego dopasowanego wariantu etykiety
    w CHV, albo None, jeśli żaden wariant nie trafił."""
    for variant in label_variants(label):
        if variant in chv_scores:
            return chv_scores[variant]
    return None


# Węzły, które trafiają w CHV z wysoką znajomością (bo ich etykieta to po
# prostu pospolite angielskie słowo), ale same w sobie nie są pojęciami
# medycznymi -- formalne kategorie organizacyjne ontologii (EFO/NCIT), nie
# treść domenowa. Ani fan-out, ani odległość od korzenia nie odróżniają ich
# automatycznie od legalnych, dużych korzeni chorobowych (patrz
# src/analyze_hierarchy_shape.py) -- to krótka, ręcznie zweryfikowana lista
# (sprawdzona przeciw wszystkim przodkom naszych 36 cech, nie zgadnięta),
# nie automatyczna reguła, więc nie generalizuje się bez przeglądu na nową
# ontologię/dziedzinę.
GENERIC_NODES: set[str] = {
    "NCIT:C43359",  # Group
    "NCIT:C43431",  # Activity
    "EFO:0001444",  # measurement
    "owl:Thing",  # uniwersalny szczyt OWL -- skrajny przypadek tego samego problemu
    "EFO:0000001",  # "experimental factor" -- własny korzeń EFO, tak samo pusty
    "NCIT:C20189",  # Property or Attribute -- 193 dzieci w zupełnie niezwiązanych
    # domenach (Dose, Efficacy, Ejection Fraction, Distance...), czysto
    # organizacyjna kategoria NCIT, nie treść medyczna
    "NCIT:C20181",  # Conceptual Entity -- 336 dzieci, w większości nie-medycznych
    # pospolitych słów (Access, Agent, Artifact, Award, Background...),
    # ten sam wzorzec co Group/Activity
    "NCIT:C70766",  # Quantitative Concept -- strukturalnie prawdziwy węzeł (np.
    # BMXWAIST/waist circumference naprawdę przez niego przechodzi), ale jako
    # etykieta wyjaśnienia jest bezużyteczny: "ryzyko cukrzycy przez Quantitative
    # Concept" nie mówi laikowi nic o TYM, JAKA wielkość jest mierzona. Inne
    # kryterium niż false-positive CHV / organizacyjny worek powyżej -- tu wadą
    # jest brak czytelności jako wyjaśnienie, a nie brak treści ontologicznej.
}

# ROZWAŻONE i ODRZUCONE (sprawdzone na danych, nie zgadnięte):
# - "Personal Attribute" (NCIT:C19332): NIE blacklistować. W przeciwieństwie do
#   powyższych, to realnie zwęża zakres -- 83 dzieci, wszystkie o cechach osoby
#   (Age, Body Mass Index, Ancestry, Birthplace...), nie przypadkowy zbiór
#   pospolitych słów. Co ważne: to BEZPOŚREDNI rodzic RIDAGEYR (Age) w naszych
#   36 cechach -- zbanowanie go odcięłoby Age od JAKIEGOKOLWIEK grupowania przy
#   każdym target_specificity, bo nie ma czym go zastąpić. To jest już granica,
#   za którą kryterium "nic mi nie mówi" przestaje mieć obiektywne uzasadnienie
#   (fałszywe trafienie w CHV / dosłowny korzeń) i staje się czysto subiektywne
#   -- a subiektywnie KAŻDy węzeł pośredni "mówi mniej" niż liść, bo o to właśnie
#   chodzi w abstrahowaniu. Pociągnięte do końca, to kryterium ucięłoby całą
#   hierarchię z powrotem do samych liści.


def is_generic_node(curie: str) -> bool:
    """Czy ten węzeł jest znaną, nie-medyczną kotwicą CHV (patrz
    GENERIC_NODES) -- jego trafienie w CHV powinno być ignorowane."""
    return curie in GENERIC_NODES


if __name__ == "__main__":
    chv_scores = load_chv_scores()
    for label in [
        "asthma",
        "Status Asthmaticus",
        "nephrolithiasis",
        "Congestive Heart Failure",
        "Abdominal Pain, CTCAE",
        "Mouse Gallbladder",
    ]:
        fam = chv_familiarity(label, chv_scores)
        print(f"{label!r:35s} -> familiarity={fam}")
