"""Analyse psycholinguistique basée sur le lexique FEEL.

Ce module propose une granularité émotionnelle fine (joie, colère, peur,
tristesse, surprise, dégoût) pour l'analyse de discours. Il peut exploiter
un fichier CSV du lexique FEEL placé dans ``dictionnaires/feel.csv``
ou s'appuyer sur un échantillon minimal inclus pour des démonstrations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

from text_utils import normaliser_espace

# Lexique minimal intégré pour fonctionner même sans fichier externe.
# Les colonnes du CSV attendu :
# - format long : "lemme" (ou "word"), "emotion", "polarity".
# - format large FEEL : "id", "word", "polarity", "joy", "fear",
#   "sadness", "anger", "surprise", "disgust" (séparateur ";").
LEXIQUE_FEEL_REDUIT = [
    {"lemme": "joie", "emotion": "joy", "polarity": "positive"},
    {"lemme": "bonheur", "emotion": "joy", "polarity": "positive"},
    {"lemme": "heureux", "emotion": "joy", "polarity": "positive"},
    {"lemme": "colère", "emotion": "anger", "polarity": "negative"},
    {"lemme": "rage", "emotion": "anger", "polarity": "negative"},
    {"lemme": "haine", "emotion": "anger", "polarity": "negative"},
    {"lemme": "peur", "emotion": "fear", "polarity": "negative"},
    {"lemme": "crainte", "emotion": "fear", "polarity": "negative"},
    {"lemme": "angoisse", "emotion": "fear", "polarity": "negative"},
    {"lemme": "tristesse", "emotion": "sadness", "polarity": "negative"},
    {"lemme": "malheur", "emotion": "sadness", "polarity": "negative"},
    {"lemme": "pleurer", "emotion": "sadness", "polarity": "negative"},
    {"lemme": "surprise", "emotion": "surprise", "polarity": "positive"},
    {"lemme": "étonnement", "emotion": "surprise", "polarity": "positive"},
    {"lemme": "imprévu", "emotion": "surprise", "polarity": "positive"},
    {"lemme": "dégoût", "emotion": "disgust", "polarity": "negative"},
    {"lemme": "écœurement", "emotion": "disgust", "polarity": "negative"},
    {"lemme": "répugnance", "emotion": "disgust", "polarity": "negative"},
]


@dataclass
class EmotionScore:
    """Score agrégé pour une émotion donnée."""

    emotion: str
    polarite: str
    occurrences: int
    poids: float
    discours: str


class FeelLexiqueErreur(RuntimeError):
    """Erreur liée au chargement ou au format du lexique FEEL."""


_DEF_CHEMIN_FEEL = Path(__file__).resolve().parent.parent / "dictionnaires" / "feel.csv"


def _normaliser_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """Uniformise les colonnes attendues du lexique FEEL.

    Le CSV peut être livré soit en format « long » (colonnes ``lemme``,
    ``emotion``, ``polarite``), soit en format « large » utilisé par le
    lexique FEEL officiel avec une colonne par émotion. Dans ce dernier cas,
    la fonction déplie les émotions présentes (valeur non nulle) et conserve
    la polarité associée.
    """

    df = df.copy()
    colonnes_equivalentes: Dict[str, str] = {
        "word": "lemme",
        "term": "lemme",
        "polarity": "polarite",
        "valence": "polarite",
    }
    df.rename(columns=colonnes_equivalentes, inplace=True)

    colonnes_emotions = ["joy", "fear", "sadness", "anger", "surprise", "disgust"]
    colonnes_large = set(colonnes_emotions).issubset(df.columns)

    if "emotion" not in df.columns and colonnes_large:
        if "lemme" not in df.columns:
            raise FeelLexiqueErreur(
                "Le lexique FEEL doit contenir une colonne 'lemme' ou 'word'."
            )
        if "polarite" not in df.columns:
            raise FeelLexiqueErreur(
                "Le lexique FEEL doit contenir une colonne 'polarity' ou 'polarite'."
            )

        df = (
            df.melt(
                id_vars=["lemme", "polarite"],
                value_vars=colonnes_emotions,
                var_name="emotion",
                value_name="score",
            )
            .loc[lambda d: d["score"].fillna(0) != 0]
            .drop(columns=["score"])
        )
    else:
        colonnes_manquantes = {"lemme", "emotion", "polarite"} - set(df.columns)
        if colonnes_manquantes:
            raise FeelLexiqueErreur(
                "Le lexique FEEL doit contenir les colonnes 'lemme', 'emotion' et 'polarite'."
            )

        df = df[["lemme", "emotion", "polarite"]]

    df["lemme"] = df["lemme"].astype(str).str.lower().str.strip()
    df["emotion"] = df["emotion"].astype(str).str.lower().str.strip()
    df["polarite"] = df["polarite"].astype(str).str.lower().str.strip()
    return df[["lemme", "emotion", "polarite"]]


def charger_lexique_feel(chemin: Optional[Path | str] = None) -> pd.DataFrame:
    """Charge le lexique FEEL depuis un CSV ou l'échantillon minimal.

    Parameters
    ----------
    chemin:
        Emplacement du CSV FEEL. Si ``None`` et si le fichier ``dictionnaires/feel.csv``
        existe, il sera utilisé. Sinon un petit lexique intégré sera retourné.
    """

    if chemin is None:
        chemin = _DEF_CHEMIN_FEEL

    if chemin and Path(chemin).exists():
        try:
            df = pd.read_csv(chemin)
            if df.shape[1] == 1 and df.columns[0].count(";") >= 2:
                df = pd.read_csv(chemin, sep=";")
        except Exception as err:  # pragma: no cover - dépendance externe
            raise FeelLexiqueErreur(
                f"Impossible de charger le lexique FEEL depuis {chemin} : {err}"
            ) from err
        return _normaliser_colonnes(df)

    return _normaliser_colonnes(pd.DataFrame(LEXIQUE_FEEL_REDUIT))


def _tokeniser_texte(texte: str) -> List[str]:
    """Découpe le texte en tokens alphanumériques et gère les accents."""

    if not texte:
        return []
    tokens = re.findall(r"[\wÀ-ÖØ-öø-ÿ'-]+", texte.lower())
    return tokens


def analyser_emotions_feel(texte: str, lexique: pd.DataFrame, discours: str) -> List[EmotionScore]:
    """Calcule la distribution des émotions FEEL pour un discours."""

    texte_norm = normaliser_espace(texte)
    tokens = _tokeniser_texte(texte_norm)
    if not tokens or lexique.empty:
        return []

    df_tokens = pd.DataFrame({"lemme": tokens})
    df_tokens["lemme"] = df_tokens["lemme"].astype(str).str.lower()
    jointure = df_tokens.merge(lexique, on="lemme", how="inner")
    if jointure.empty:
        return []

    tot_tokens = len(tokens)
    agregat = (
        jointure.groupby(["emotion", "polarite"], as_index=False)
        .size()
        .rename(columns={"size": "occurrences"})
    )
    agregat["poids"] = agregat["occurrences"] / float(tot_tokens)
    return [
        EmotionScore(
            emotion=row["emotion"],
            polarite=row["polarite"],
            occurrences=int(row["occurrences"]),
            poids=float(row["poids"]),
            discours=discours,
        )
        for _, row in agregat.iterrows()
    ]


def _scores_en_dataframe(scores: Iterable[EmotionScore]) -> pd.DataFrame:
    """Transforme les scores en DataFrame."""

    data = [s.__dict__ for s in scores]
    if not data:
        return pd.DataFrame(
            columns=["emotion", "polarite", "occurrences", "poids", "discours"]
        )
    return pd.DataFrame(data)


def _afficher_intro_methodologie():
    """Texte introductif sur le lexique FEEL."""

    st.markdown("### FEEL — French Expanded Emotion Lexicon")
    st.markdown(
        "Le lexique FEEL propose une catégorisation émotionnelle fine (joie, colère, peur,"
        " tristesse, surprise, dégoût) plutôt qu'une simple polarité positif/négatif."
    )
    st.markdown(
        "Le script calcule la fréquence des lexèmes présents dans le texte et leur poids"
        " relatif par rapport au nombre total de mots. Chargez un CSV 'feel.csv' dans"
        " le dossier `dictionnaires/` pour une couverture maximale ou utilisez le"
        " lexique réduit intégré pour tester rapidement."
    )


def _visualiser_scores(df: pd.DataFrame, titre: str):
    """Affiche un histogramme des émotions détectées."""

    if df.empty:
        st.info("Aucune émotion FEEL détectée dans ce discours.")
        return

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("emotion:N", sort="-x", title="Émotion"),
            x=alt.X("occurrences:Q", title="Occurrences"),
            color=alt.Color("polarite:N", title="Polarité"),
            tooltip=["emotion", "polarite", "occurrences", alt.Tooltip("poids:Q", format=".2%")],
        )
        .properties(title=titre)
    )
    st.altair_chart(chart, use_container_width=True)


def render_feel_tab(
    texte_discours_1: str,
    texte_discours_2: str,
    nom_discours_1: str,
    nom_discours_2: str,
):
    """Rendu Streamlit pour l'onglet FEEL."""

    _afficher_intro_methodologie()

    if not texte_discours_1.strip() and not texte_discours_2.strip():
        st.info(
            "Aucun discours fourni. Ajoutez du texte pour lancer l'analyse psycholinguistique FEEL."
        )
        return

    try:
        lexique_feel = charger_lexique_feel()
    except FeelLexiqueErreur as err:
        st.error(str(err))
        return

    discours_disponibles: Dict[str, str] = {}
    if texte_discours_1.strip():
        discours_disponibles[nom_discours_1] = texte_discours_1
    if texte_discours_2.strip():
        discours_disponibles[nom_discours_2] = texte_discours_2

    onglets = st.tabs(list(discours_disponibles.keys()))
    for tab, (nom, contenu) in zip(onglets, discours_disponibles.items()):
        with tab:
            st.markdown(f"#### {nom}")
            scores = analyser_emotions_feel(contenu, lexique_feel, nom)
            df_scores = _scores_en_dataframe(scores)
            if df_scores.empty:
                st.info("Aucune correspondance FEEL trouvée dans ce discours.")
                continue

            st.dataframe(
                df_scores.sort_values("occurrences", ascending=False),
                use_container_width=True,
            )
            _visualiser_scores(df_scores, titre=f"Distribution émotionnelle — {nom}")
