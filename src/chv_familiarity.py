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
