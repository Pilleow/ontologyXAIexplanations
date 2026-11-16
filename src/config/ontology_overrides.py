# Ręczne poprawki ponad automatycznym mapowaniem ontologicznym.

# Węzły zbyt ogólne/organizacyjne, przez które wspinaczka nie powinna
# przechodzić (patrz src/fix_mappings.py i historia commitów po uzasadnienia).
GENERIC_NODES: set[str] = {
    "NCIT:C43359",  # Group
    "NCIT:C43431",  # Activity
    "EFO:0001444",  # measurement
    "owl:Thing",
    "EFO:0000001",  # experimental factor
    "NCIT:C20189",  # Property or Attribute
    "NCIT:C20181",  # Conceptual Entity
    "NCIT:C70766",  # Quantitative Concept
}

# zmienna -> (CURIE, etykieta), gdy automatyczne mapowanie (text2term) było
# błędne/puste -- zweryfikowane przez src/fix_mappings.py.
MANUAL_OVERRIDES: dict[str, tuple[str, str]] = {
    "CDQ001": ("NCIT:C38665", "Chest Pain"),
    "BPQ080": ("NCIT:C37967", "Hypercholesterolemia"),
    "MCQ160E": ("NCIT:C27996", "Myocardial Infarction"),
    "MCQ160N": ("EFO:0004274", "gout"),
    "RIDAGEYR": ("NCIT:C25150", "Age"),
    "PAQ650": ("NCIT:C39773", "Strenuous Exercise"),
    "PAQ665": ("NCIT:C39772", "Moderate Exercise"),
    "RIDRETH3": ("NCIT:C17049", "Race"),
    "INDFMPIR": ("NCIT:C41150", "Income"),
    "BPXSY1": ("NCIT:C25298", "Systolic Blood Pressure"),
    "BPXDI1": ("NCIT:C25299", "Diastolic Blood Pressure"),
    "PAQ605": ("NCIT:C39773", "Strenuous Exercise"),
    "SMQ020": ("NCIT:C18270", "Cigarette Smoking"),
    "CDQ010": ("EFO:0009727", "shortness of breath"),
    "DPQ010": ("NCIT:C3673", "Anhedonia"),
    "DPQ020": ("NCIT:C117199", "Depressed Mood"),
    "KIQ029": ("EFO:0004253", "nephrolithiasis"),
    "OSQ060": ("NCIT:C3298", "Osteoporosis"),
}
