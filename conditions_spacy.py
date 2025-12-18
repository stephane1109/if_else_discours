"""
Détection des structures conditionnelles via spaCy.

La fonction principale retourne un DataFrame (id_phrase, type, connecteur,
commentaire, phrase) afin d’être affiché directement dans Streamlit. La logique
reste volontairement simple : on cherche les locutions présentes dans les
dictionnaires (condition/then/else) et on les note pour chaque phrase analysée
par spaCy.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

import pandas as pd

EMPTY_COLS = ["id_phrase", "type", "connecteur", "commentaire", "phrase"]


def _normaliser_locutions(locutions: Iterable[str]) -> List[str]:
    """Nettoie et dédoublonne les locutions en minuscules pour faciliter les recherches."""
    seen = set()
    cleaned: List[str] = []
    for loc in locutions:
        loc_clean = loc.strip().lower()
        if not loc_clean or loc_clean in seen:
            continue
        seen.add(loc_clean)
        cleaned.append(loc_clean)
    # On trie par longueur décroissante pour privilégier les locutions les plus longues
    return sorted(cleaned, key=len, reverse=True)


def _trouver_locution(phrase: str, locutions: Sequence[str]) -> str:
    """Retourne la première locution trouvée (sensibilité à la casse ignorée)."""
    if not phrase:
        return ""
    phrase_norm = phrase.lower()
    for loc in locutions:
        if re.search(rf"\\b{re.escape(loc)}\\b", phrase_norm):
            return loc
    return ""


def analyser_conditions_spacy(
    texte: str,
    nlp,
    cond_terms: Sequence[str],
    alors_terms: Sequence[str],
    alt_terms: Sequence[str],
) -> pd.DataFrame:
    """
    Analyse les phrases avec spaCy et repère la présence de connecteurs
    conditionnels (if/then/else).

    Args:
        texte: Texte à analyser.
        nlp: Pipeline spaCy déjà chargé.
        cond_terms: Liste/ensemble des déclencheurs de condition ("si", "à condition que", …).
        alors_terms: Liste/ensemble des déclencheurs de conséquence ("alors", "donc", …).
        alt_terms: Liste/ensemble des déclencheurs d'alternative ("sinon", "dans le cas inverse", …).

    Returns:
        pd.DataFrame avec les colonnes id_phrase, type, connecteur, commentaire, phrase.
    """
    if not texte or nlp is None:
        return pd.DataFrame(columns=EMPTY_COLS)

    cond_locutions = _normaliser_locutions(cond_terms)
    alors_locutions = _normaliser_locutions(alors_terms)
    alt_locutions = _normaliser_locutions(alt_terms)

    doc = nlp(texte)
    enregs = []

    for pid, sent in enumerate(doc.sents, start=1):
        phrase_txt = sent.text.strip()
        if not phrase_txt:
            continue

        found_cond = _trouver_locution(phrase_txt, cond_locutions)
        found_then = _trouver_locution(phrase_txt, alors_locutions)
        found_alt = _trouver_locution(phrase_txt, alt_locutions)

        if not any((found_cond, found_then, found_alt)):
            continue

        if found_cond:
            enregs.append(
                {
                    "id_phrase": pid,
                    "type": "CONDITION",
                    "connecteur": found_cond,
                    "commentaire": "Segment conditionnel détecté",
                    "phrase": phrase_txt,
                }
            )
        if found_then:
            enregs.append(
                {
                    "id_phrase": pid,
                    "type": "ALORS",
                    "connecteur": found_then,
                    "commentaire": "Conséquence repérée dans la phrase",
                    "phrase": phrase_txt,
                }
            )
        if found_alt:
            enregs.append(
                {
                    "id_phrase": pid,
                    "type": "ALTERNATIVE",
                    "connecteur": found_alt,
                    "commentaire": "Alternative/sinon détecté",
                    "phrase": phrase_txt,
                }
            )

    df = pd.DataFrame(enregs, columns=EMPTY_COLS)
    if not df.empty:
        df = df.sort_values(["id_phrase", "type", "connecteur"]).reset_index(drop=True)
    return df
