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

from chv_familiarity import chv_familiarity, is_generic_node, load_chv_scores
from term_specificity import load_parent_map

META_DIR = Path(__file__).resolve().parents[1] / "external" / "NHANES-metadata"
VARIABLE_MAPPINGS = META_DIR / "ontology-mappings" / "nhanes_variables_mappings.tsv"
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
    # "high cholesterol" -> "Hypercholesterolemia" (NCIT:C37967). Rodzic to
    # "Abnormal Blood Chemistry and Hematology Test Result" -- realny,
    # zrozumiały koncept, w przeciwieństwie do poprzedniej wersji tej
    # poprawki ("Cholesterol High, CTCAE"), która wspinała się przez
    # żargon protokołów badań klinicznych ("Adverse Event", "CTCAE by
    # Body System"). Oryginalnie brak mapowania w ogóle dla tego cyklu.
    "BPQ080": ("NCIT:C37967", "Hypercholesterolemia"),
    # "heart attack" nie ma dobrego kandydata w text2term (najlepszy:
    # "Heart" 0.673, zupełnie ogólny). Używamy więc CURIE, które upstream
    # poprawnie przypisał siostrzanej zmiennej o wieku rozpoznania tego
    # samego schorzenia (MCD180E), zamiast zgadywać od nowa.
    "MCQ160E": ("NCIT:C27996", "Myocardial Infarction"),
    # "gout" -> gout/Gout (wynik 0.996, prawie idealny). Oryginalnie brak
    # mapowania w ogóle dla tej (nie wiekowej) zmiennej.
    "MCQ160N": ("EFO:0004274", "gout"),
    # SourceTerm: "Age in years at screening" -- text2term dopasował do
    # "Screening Study" (NCIT:C71485, wynik 0.739), gubiąc "age" na rzecz
    # "screening". Istnieje czysty koncept "Age" (NCIT:C25150, rodzic
    # "Personal Attribute" -- ta sama gałąź co RIAGENDR/"Gender", więc wiek
    # i płeć będą się sensownie grupować razem wyżej). RIDAGEYR jest #1 wg
    # globalnego |SHAP| w tym modelu, dwukrotnie przed kolejną cechą.
    "RIDAGEYR": ("NCIT:C25150", "Age"),
    # SourceTerm: "Vigorous recreational activities" -- text2term dopasował
    # do niezwiązanego NCIT:C148226 "Extent Breast Cancer Treatment
    # Disrupted Recreational Activities" (pytanie o zakłócenie aktywności
    # przez leczenie raka piersi, nie o poziom aktywności w ogóle).
    # "Strenuous Exercise" (NCIT:C39773) to rodzeństwo "Moderate Exercise"
    # (NCIT:C39772, już używane przez PAQ620) w tej samej gałęzi "Exercise".
    "PAQ650": ("NCIT:C39773", "Strenuous Exercise"),
    # SourceTerm: "Moderate recreational activities" -- ten sam błąd co
    # PAQ650. Mapujemy na ten sam koncept co PAQ620 ("Moderate Exercise",
    # NCIT:C39772) -- ten sam poziom intensywności, inny kontekst pytania
    # (rekreacja vs praca), ale to ten sam koncept ontologiczny.
    "PAQ665": ("NCIT:C39772", "Moderate Exercise"),
    # "race and ethnicity" -> text2term dał "American Ethnicity" (wynik
    # 0.644), ale to okazało się BŁĘDNE: ten konkretny termin to wąska
    # podgrupa etniczna ("osoba pochodząca z Ameryk", rodzeństwo "Han
    # Chinese"/"Sephardic Jew" w tej samej gałęzi), nie ogólna kategoria
    # rasy/pochodzenia, jaką jest RIDRETH3. Właściwy, generyczny koncept
    # to "Race" (NCIT:C17049, rodzic "Demographic Group"). Oryginalnie
    # brak mapowania w ogóle.
    "RIDRETH3": ("NCIT:C17049", "Race"),
    # "poverty income ratio" -> Income (wynik 0.593, CHV familiarity
    # 0.775), rodzic "Socioeconomic Factors" -- ta sama gałąź co
    # DMDEDUC2/"Education Level". Oryginalnie brak mapowania w ogóle.
    "INDFMPIR": ("NCIT:C41150", "Income"),
    # "systolic blood pressure" -> Systolic Blood Pressure (wynik 0.988,
    # niemal idealny), rodzic "Blood Pressure". Oryginalnie brak mapowania
    # w ogóle.
    "BPXSY1": ("NCIT:C25298", "Systolic Blood Pressure"),
    # "diastolic blood pressure" -> Diastolic Blood Pressure (wynik 0.983).
    # Oryginalnie brak mapowania w ogóle.
    "BPXDI1": ("NCIT:C25299", "Diastolic Blood Pressure"),
    # SourceTerm: "Vigorous work activity" -- ta sama intensywność co
    # PAQ650 (vigorous recreational), inny kontekst (praca). Mapujemy na
    # ten sam koncept dla spójności. Oryginalnie brak mapowania w ogóle.
    "PAQ605": ("NCIT:C39773", "Strenuous Exercise"),
    # "cigarette smoking" -> Cigarette Smoking (wynik 0.975), rodzic
    # "Tobacco Smoking" -- realny koncept, w przeciwieństwie do
    # oryginalnego mapowania na etykietę pytania ("Smoking Question"),
    # które wspinało się dalej przez "Clinical or Research Assessment
    # Question" -- tę samą generyczną taksonomię pytań co CDQ010/DPQ*.
    "SMQ020": ("NCIT:C18270", "Cigarette Smoking"),
    # "shortness of breath" -> shortness of breath (EFO:0009727, wynik
    # 0.976) -- realny symptom, zamiast wspinać się przez "Have Symptom
    # or Condition Question" -> generyczną taksonomię pytań ankietowych.
    "CDQ010": ("EFO:0009727", "shortness of breath"),
    # "anhedonia" (kliniczny termin na "mało zainteresowania/przyjemności")
    # -> Anhedonia (NCIT:C3673, wynik 0.970), rodzic "Mental and
    # Behavioral Signs and Symptoms" -- ta sama gałąź co DPQ020, więc oba
    # pytania PHQ-9 będą się sensownie grupować.
    "DPQ010": ("NCIT:C3673", "Anhedonia"),
    # "depressed mood" -> Depressed Mood (NCIT:C117199, wynik 0.978), ta
    # sama gałąź "Mental and Behavioral Signs and Symptoms" co DPQ010.
    "DPQ020": ("NCIT:C117199", "Depressed Mood"),
    # SourceTerm: "Pass kidney stone in past 12 months" -- text2term nie
    # ma dobrego kandydata dla "kidney stone" (najlepszy: "Kidney" 0.749,
    # "Rat Kidney" 0.672 -- zupełnie nieistotne). Oryginalne mapowanie
    # ("Renal Stone", NCIT:C26807) wspinało się przez "Material"/
    # "Specimen"/"Biospecimen" -- traktując kamień jak próbkę
    # laboratoryjną, nie schorzenie. Używamy tego samego konceptu
    # choroby co KIQ026 (ten sam temat -- kamienie nerkowe).
    "KIQ029": ("EFO:0004253", "nephrolithiasis"),
    # "osteoporosis" -> Osteoporosis (NCIT:C3298, wynik 0.983), rodzic
    # "Metabolic Bone Disorder" -- realny koncept, zamiast poprzedniej
    # wersji tej poprawki ("Osteoporosis, CTCAE 3.0"), która wspinała się
    # przez żargon protokołów badań klinicznych ("Adverse Event").
    "OSQ060": ("NCIT:C3298", "Osteoporosis"),
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


def ancestor_chains(
    leaf_curies: set[str], chv_scores: dict[str, float] | None = None
) -> dict[str, list[str]]:
    """Dla każdego liścia: zdeterminowany łańcuch od liścia (chain[0]) do
    korzenia (chain[-1], węzła bez dalszego rodzica). Gdy węzeł ma kilku
    rodziców (polihierarchia EFO/NCIT), wybierany jest ten, którego gałąź
    dociera do NAJBLIŻSZEJ kotwicy CHV -- NIE "pierwszy znaleziony w
    pliku" jak poprzednio. To miało znaczenie w praktyce: dla zawału serca
    (Myocardial Infarction) "pierwszy znaleziony" prowadził przez
    "Muscle Disorder"/"Musculoskeletal Disorder" (mięsień sercowy
    potraktowany jak zwykła tkanka mięśniowa), mimo że w tych samych danych
    istniała właściwa gałąź "Heart Disorder"/"Cardiovascular Disorder".

    Dla każdego węzła liczymy nearest_anchor = (liczba kroków do najbliższej
    kotwicy CHV, -familiarity w tej kotwicy): 0 kroków, jeśli własna
    etykieta trafia w CHV, inaczej 1 + minimum (po liczbie kroków) spośród
    rodziców. Przy wspinaczce, na każdej rozgałęzieniu wybieramy rodzica o
    NAJMNIEJSZEJ liczbie kroków do kotwicy (remisy: wyższa familiarity,
    potem alfabetycznie po CURIE). Dlaczego najbliższa, nie najwyższa
    gdziekolwiek: prawie każda gałąź i tak ostatecznie trafia w ten sam,
    bardzo ogólny węzeł blisko korzenia (np. "Disease, Disorder or
    Finding", familiarity≈1.0 -- to po prostu słowo "choroba"), więc
    branie maksimum "gdziekolwiek" nie rozróżniało gałęzi wcale."""
    if chv_scores is None:
        chv_scores = load_chv_scores()
    full_parent_map = load_parent_map()

    per_leaf_levels: dict[str, list[list[str]]] = {}
    all_nodes: set[str] = set()
    for leaf in leaf_curies:
        dist_from_leaf = {leaf: 0}
        levels = [[leaf]]
        frontier = [leaf]
        while frontier:
            next_frontier = []
            for node in frontier:
                for parent in full_parent_map.get(node, ()):
                    if parent not in dist_from_leaf:
                        dist_from_leaf[parent] = dist_from_leaf[node] + 1
                        next_frontier.append(parent)
            if next_frontier:
                levels.append(next_frontier)
            frontier = next_frontier
        per_leaf_levels[leaf] = levels
        all_nodes.update(dist_from_leaf)

    label_of = load_labels(all_nodes)

    # Dla każdego węzła: (liczba kroków do najbliższej kotwicy CHV w górę,
    # -familiarity w tej kotwicy). Mniejszy klucz (leksykograficznie) = lepsza
    # gałąź -- NAJBLIŻSZA kotwica wygrywa, nie najwyższa znaleziona
    # gdziekolwiek. To jest kluczowe: prawie każda gałąź EFO/NCIT i tak
    # ostatecznie trafia w ten sam, bardzo ogólny węzeł blisko korzenia
    # (np. "Disease, Disorder or Finding", familiarity≈1.0 -- to po prostu
    # słowo "choroba"), więc branie maksimum "gdziekolwiek" nie rozróżnia
    # gałęzi w ogóle -- liczy się to, co jest NAJBLIŻEJ.
    nearest_anchor: dict[str, tuple[int, float]] = {}
    no_anchor = (10**9, 0.0)
    for leaf, levels in per_leaf_levels.items():
        for level in reversed(levels):
            for node in level:
                if node in nearest_anchor:
                    continue
                label = label_of.get(node)
                fam = chv_familiarity(label, chv_scores) if label and not is_generic_node(node) else None
                if fam is not None:
                    nearest_anchor[node] = (0, -fam)
                else:
                    parents = full_parent_map.get(node, ())
                    candidates = [nearest_anchor[p] for p in parents if p in nearest_anchor]
                    best = min(candidates) if candidates else no_anchor
                    nearest_anchor[node] = (best[0] + 1, best[1]) if candidates else no_anchor

    chains: dict[str, list[str]] = {}
    for leaf in per_leaf_levels:
        chain = [leaf]
        current = leaf
        while True:
            parents = full_parent_map.get(current, ())
            if not parents:
                break
            next_node = min(sorted(parents), key=lambda p: nearest_anchor.get(p, no_anchor))
            # Węzeł z GENERIC_NODES (np. Group, Activity, owl:Thing) albo
            # bez żadnej etykiety w ogóle (BFO:*/IAO:0000030 -- pokazalibyśmy
            # użytkownikowi gołe CURIE) nic nie wnosi -- a skoro wspinamy
            # się w stronę coraz bardziej abstrakcyjnych pojęć, wszystko
            # POWYŻEJ niego jest co najmniej tak samo bezużyteczne. Ucinamy
            # łańcuch tutaj: ostatni dodany węzeł staje się efektywnym
            # korzeniem tej ścieżki -- nieważne, jak niski
            # target_specificity, wspinaczka nie pójdzie dalej.
            if is_generic_node(next_node) or label_of.get(next_node) is None:
                break
            chain.append(next_node)
            current = next_node
        chains[leaf] = chain

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
        fam = chv_familiarity(label, chv_scores) if label and not is_generic_node(curie) else None
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
    zdeterminowanym, sterowanym przez CHV łańcuchu (patrz ancestor_chains)
    aż specificity_score (najpierw z CHV przez dopasowanie tekstowe
    etykiety, w przeciwnym razie strukturalny) spadnie do/poniżej
    `target_specificity` (1.0 -> zostaje na liściu, 0.0 -> idzie do
    korzenia)."""
    chv_scores = load_chv_scores()
    chains = ancestor_chains(leaf_curies, chv_scores)
    label_of = load_labels({c for chain in chains.values() for c in chain})

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
    rodzica (is_a) wg tej samej, sterowanej przez CHV ścieżki co
    ancestor_chains, używana jako klucz grupowania jeden poziom wyżej."""
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
