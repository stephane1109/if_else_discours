"""Analyse de co-occurrences à partir du texte fourni.

Ce module propose un rendu Streamlit avec tableau, graphique Altair et nuage de mots
pour visualiser les co-occurrences calculées phrase par phrase.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from itertools import combinations
from typing import Iterable, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

try:  # pragma: no cover - dépendance optionnelle à l'import
    import spacy
    from spacy.language import Language
except ImportError:  # pragma: no cover - spaCy non installé
    spacy = None
    Language = None

try:  # pragma: no cover - spaCy non installé
    from spacy.lang.fr.stop_words import STOP_WORDS as SPACY_STOP_WORDS
except ImportError:  # pragma: no cover - spaCy non installé
    SPACY_STOP_WORDS = set()


_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")


@lru_cache(maxsize=1)
def _charger_modele_spacy() -> Optional["Language"]:
    """Charge le modèle spaCy français en le mettant en cache."""

    if spacy is None:
        return None
    try:
        return spacy.load("fr_core_news_sm")
    except OSError:
        # Le modèle n'est pas disponible localement
        return None
    except Exception:
        # Toute autre erreur (par ex. incompatibilité) -> désactive le filtrage
        return None


def _segmenter_en_phrases(texte: str) -> List[str]:
    """Segmente le texte en phrases en se basant sur la ponctuation forte."""
    if not texte:
        return []
    morceaux = re.split(r"(?<=[\.\!\?\:\;])\s+", texte)
    return [m.strip() for m in morceaux if m and m.strip()]


def _extraire_mots(
    phrase: str,
    *,
    longueur_min: int = 2,
    modele_spacy: Optional["Language"] = None,
) -> List[str]:
    """Extrait des mots en minuscule, filtrés sur la longueur minimale."""
    if not phrase:
        return []

    if modele_spacy is None:
        modele_spacy = _charger_modele_spacy()

    if modele_spacy is not None:
        doc = modele_spacy(phrase)
        tokens = []
        for token in doc:
            if not token.is_alpha:
                continue
            if len(token.text) < longueur_min:
                continue
            if token.is_stop:
                continue
            lemme = token.lemma_.lower() if token.lemma_ else token.text.lower()
            tokens.append(lemme)
        if tokens:
            return tokens

    mots = [m.lower() for m in _WORD_PATTERN.findall(phrase)]
    stopwords = SPACY_STOP_WORDS if SPACY_STOP_WORDS else set()
    return [m for m in mots if len(m) >= longueur_min and m not in stopwords]


def _generer_cooccurrences(
    phrases: Iterable[str],
    longueur_min: int = 2,
    modele_spacy: Optional["Language"] = None,
) -> Counter[str]:
    """Compte les co-occurrences (paires ordonnées lexicographiquement) par phrase."""
    compteurs: Counter[str] = Counter()
    for phrase in phrases:
        tokens = sorted(
            set(
                _extraire_mots(
                    phrase,
                    longueur_min=longueur_min,
                    modele_spacy=modele_spacy,
                )
            )
        )
        if len(tokens) < 2:
            continue
        for mot1, mot2 in combinations(tokens, 2):
            pair = f"{mot1}_{mot2}"
            compteurs[pair] += 1
    return compteurs


def calculer_table_cooccurrences(texte: str, longueur_min: int = 2) -> pd.DataFrame:
    """Retourne un DataFrame des co-occurrences triées par fréquence décroissante."""
    phrases = _segmenter_en_phrases(texte)
    modele_spacy = _charger_modele_spacy()
    compteur = _generer_cooccurrences(
        phrases,
        longueur_min=longueur_min,
        modele_spacy=modele_spacy,
    )
    if not compteur:
        return pd.DataFrame(columns=["mot1", "mot2", "pair", "occurrences"])

    donnees = []
    for pair, freq in compteur.most_common():
        mot1, mot2 = pair.split("_", 1)
        donnees.append({
            "mot1": mot1,
            "mot2": mot2,
            "pair": pair,
            "occurrences": freq,
        })

    df = pd.DataFrame(donnees)
    df["occurrences"] = pd.to_numeric(df["occurrences"], errors="coerce").fillna(0).astype(int)
    return df


def _graphique_barres_cooccurrences(df: pd.DataFrame, top_n: int) -> alt.Chart | None:
    """Construit un graphique à barres Altair pour les co-occurrences les plus fréquentes."""
    if df.empty:
        return None

    top_df = df.head(top_n).copy()
    hauteur = max(300, 18 * len(top_df))
    chart = (
        alt.Chart(top_df)
        .mark_bar()
        .encode(
            x=alt.X("occurrences:Q", title="Occurrences"),
            y=alt.Y("pair:N", sort="-x", title="Paire de mots"),
            color=alt.Color("occurrences:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=[
                alt.Tooltip("mot1:N", title="Mot 1"),
                alt.Tooltip("mot2:N", title="Mot 2"),
                alt.Tooltip("occurrences:Q", title="Occurrences"),
            ],
        )
        .properties(height=hauteur)
    )
    return chart


def _nuage_de_mots(df: pd.DataFrame, max_mots: int) -> alt.Chart | None:
    """Construit un nuage de mots (mise en grille) pour visualiser les co-occurrences."""
    if df.empty:
        return None

    data = df.head(max_mots).copy()
    data["rang"] = range(len(data))
    if data.empty:
        return None

    nb_cols = max(4, int(math.sqrt(len(data))) or 1)
    data["colonne"] = data["rang"] % nb_cols
    data["ligne"] = -(data["rang"] // nb_cols)

    chart = (
        alt.Chart(data)
        .mark_text(baseline="middle", align="center")
        .encode(
            x=alt.X("colonne:Q", axis=None),
            y=alt.Y("ligne:Q", axis=None),
            text=alt.Text("pair:N"),
            size=alt.Size("occurrences:Q", scale=alt.Scale(range=[10, 80])),
            color=alt.Color("occurrences:Q", scale=alt.Scale(scheme="plasma"), legend=None),
            tooltip=[
                alt.Tooltip("pair:N", title="Co-occurrence"),
                alt.Tooltip("occurrences:Q", title="Occurrences"),
            ],
        )
        .configure_view(strokeWidth=0)
        .properties(width=700, height=400)
    )
    return chart


def render_cooccurrences_tab(texte_source: str) -> None:
    """Affiche l'onglet Streamlit consacré aux co-occurrences."""
    st.subheader("Co-occurrences par phrase")

    if not texte_source or not texte_source.strip():
        st.info("Saisissez ou chargez un texte pour analyser les co-occurrences.")
        return

    longueur_min = st.slider(
        "Longueur minimale des mots (en caractères)",
        min_value=1,
        max_value=6,
        value=2,
        help="Les mots plus courts que cette valeur sont ignorés pour stabiliser les co-occurrences.",
    )

    modele_spacy = _charger_modele_spacy()
    if modele_spacy is None:
        st.warning(
            "Le modèle spaCy 'fr_core_news_sm' n'a pas pu être chargé. "
            "Les stopwords ne seront pas filtrés."
        )

    df_cooc = calculer_table_cooccurrences(texte_source, longueur_min=longueur_min)

    if df_cooc.empty:
        st.info("Aucune co-occurrence n'a été détectée avec les paramètres actuels.")
        return

    max_occ = int(df_cooc["occurrences"].max())
    filtre_occ = st.slider(
        "Occurrences minimales",
        min_value=1,
        max_value=max_occ,
        value=1,
        help="Filtre les co-occurrences en fonction du nombre de phrases où elles apparaissent.",
    )
    df_filtre = df_cooc[df_cooc["occurrences"] >= filtre_occ]

    if df_filtre.empty:
        st.info("Aucune co-occurrence ne correspond au seuil minimal sélectionné.")
        return

    st.caption(
        "Les co-occurrences sont calculées phrase par phrase : deux mots co-occurent si "
        "ils apparaissent dans la même phrase."
    )

    st.dataframe(df_filtre, use_container_width=True, hide_index=True)

    top_default = min(20, len(df_filtre))
    max_top = max(5, len(df_filtre))
    if top_default < 5:
        top_default = max_top
    top_n = st.number_input(
        "Nombre de co-occurrences à afficher dans le graphique",
        min_value=5,
        max_value=max_top,
        value=top_default,
        step=1,
    )

    chart = _graphique_barres_cooccurrences(df_filtre, int(top_n))
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Nuage de mots des co-occurrences")
    max_mots_max = max(5, len(df_filtre))
    max_mots = st.slider(
        "Nombre de co-occurrences dans le nuage de mots",
        min_value=5,
        max_value=max_mots_max,
        value=min(30, max_mots_max),
        step=1,
    )

    word_cloud_chart = _nuage_de_mots(df_filtre, int(max_mots))
    if word_cloud_chart is not None:
        st.altair_chart(word_cloud_chart, use_container_width=True)
    else:
        st.info("Impossible de générer le nuage de mots avec les paramètres sélectionnés.")
