"""Analyse de sentiments basée sur CamemBERT pour les discours."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from text_utils import normaliser_espace, segmenter_en_phrases


SENTIMENT_MODEL = "cmarkea/distilcamembert-base-sentiment"


@dataclass
class SentimentResult:
    """Représentation d'un score de sentiment par phrase."""

    id_phrase: int
    texte: str
    label: str
    score: float
    valence: float


@st.cache_resource(show_spinner=False)
def _charger_pipeline_sentiment(modele: str = SENTIMENT_MODEL):
    """Charge la pipeline Hugging Face pour l'analyse de sentiments."""

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(modele)
        model = AutoModelForSequenceClassification.from_pretrained(modele)
        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
        )
    except Exception as err:  # pragma: no cover - téléchargement réseau
        raise RuntimeError(
            "Impossible de charger le modèle de sentiment Hugging Face"
            f" '{modele}': {err}"
        ) from err


def _label_to_valence(label: str) -> int:
    """Associe une valence numérique à un label textuel."""

    label_low = label.lower()
    if "pos" in label_low:
        return 1
    if "neg" in label_low:
        return -1
    return 0


def analyser_sentiments_discours(
    texte: str,
    pipeline_sentiment,
) -> List[SentimentResult]:
    """Retourne les scores de sentiment pour chaque phrase du discours."""

    texte_norm = normaliser_espace(texte)
    phrases = segmenter_en_phrases(texte_norm) if texte_norm else []
    if not phrases:
        return []

    predictions = pipeline_sentiment(phrases, truncation=True)
    resultats: List[SentimentResult] = []
    for idx, (phrase, pred) in enumerate(zip(phrases, predictions), start=1):
        label_pred = pred.get("label", "neutral")
        proba = float(pred.get("score", 0.0))
        valence = _label_to_valence(label_pred) * proba
        resultats.append(
            SentimentResult(
                id_phrase=idx,
                texte=phrase,
                label=label_pred,
                score=proba,
                valence=valence,
            )
        )
    return resultats


def _construire_dataframe(resultats: List[SentimentResult]) -> pd.DataFrame:
    """Convertit les résultats en DataFrame et ajoute un lissage."""

    if not resultats:
        return pd.DataFrame(
            columns=["id_phrase", "texte", "label", "score", "valence", "valence_lissee"]
        )

    df = pd.DataFrame([r.__dict__ for r in resultats])
    return df


def _ajouter_lissage(df: pd.DataFrame, fenetre: int) -> pd.DataFrame:
    """Calcule la moyenne glissante de la valence."""

    if df.empty:
        return df
    df = df.copy()
    df["valence_lissee"] = df["valence"].rolling(
        window=fenetre, min_periods=1, center=True
    ).mean()
    return df


def _afficher_intro_methodologie():
    """Affiche le texte méthodologique demandé."""

    st.markdown("### ASentiments")
    st.markdown(
        "La méthode \"State of the Art\" (Hugging Face / CamemBERT)\n"
        "Cette méthode utilise des Transformers (comme BERT, pour le français : CamemBERT ou FlauBERT)"
        " qui comprennent le contexte (ex: \"C'est pas terrible\" est détecté comme négatif,"
        " alors qu'un dictionnaire simple pourrait être confus par \"terrible\")."
    )
    st.markdown(
        "**Le Principe : La Courbe de Valence Émotionnelle**\n"
        "Pour construire ce graph, le script doit attribuer un score à chaque phrase (ou segment de texte) :\n"
        "- Axe X (Temps) : Le déroulement du discours (du début à la fin).\n"
        "- Axe Y (Valence) : L'axe émotionnel.\n"
        "    - Haut (+) : Chance, Espoir, Solution, Joie, Force.\n"
        "    - Bas (-) : Malchance, Péril, Problème, Tristesse, Faiblesse.\n\n"
        "Les données brutes (phrase par phrase) donnent un graph illisible (du \"bruit\")."
        " Le secret est d'utiliser une moyenne glissante (smoothing) sur 5 ou 10 phrases pour"
        " voir apparaître la courbe réelle."
    )
    st.markdown(
        "**Les Formes Géométriques Types (Visualisation) - exemples :**\n"
        "- La Forme \"Man in a Hole\" (L'Homme dans le trou) — courbe en V ou en U.\n"
        "- La Forme \"Icarus\" (La Tragédie) — courbe en V inversé (Montée puis Chute).\n"
        "- La Forme \"Rags to Riches\" (L'Ascension / La Rampe)."
    )


def _visualiser_courbe(df: pd.DataFrame, titre: str):
    """Affiche la courbe de valence émotionnelle."""

    if df.empty:
        st.info("Aucune phrase à afficher pour ce discours.")
        return

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("id_phrase:Q", title="Temps (phrases)"),
            y=alt.Y("valence_lissee:Q", title="Valence (moyenne glissante)"),
            tooltip=["id_phrase", "valence", "valence_lissee", "label", "texte"],
        )
        .properties(title=titre)
    )
    st.altair_chart(chart, use_container_width=True)


def render_sentiments_tab(
    texte_discours_1: str,
    texte_discours_2: str,
    nom_discours_1: str,
    nom_discours_2: str,
):
    """Rendu Streamlit pour l'onglet d'analyse de sentiments."""

    _afficher_intro_methodologie()

    if not texte_discours_1.strip() and not texte_discours_2.strip():
        st.info("Aucun discours fourni. Ajoutez du texte pour lancer l'analyse de sentiments.")
        return

    try:
        pipe_sentiment = _charger_pipeline_sentiment()
    except Exception as err:
        st.error(str(err))
        return

    fenetre = st.slider(
        "Taille de la moyenne glissante (en nombre de phrases)",
        min_value=3,
        max_value=15,
        value=5,
        step=1,
    )

    discours_disponibles: Dict[str, str] = {}
    if texte_discours_1.strip():
        discours_disponibles[nom_discours_1] = texte_discours_1
    if texte_discours_2.strip():
        discours_disponibles[nom_discours_2] = texte_discours_2

    onglets = st.tabs(list(discours_disponibles.keys()))
    for tab, (nom, contenu) in zip(onglets, discours_disponibles.items()):
        with tab:
            st.markdown(f"#### {nom}")
            resultats = analyser_sentiments_discours(contenu, pipe_sentiment)
            df_res = _ajouter_lissage(_construire_dataframe(resultats), fenetre)
            if df_res.empty:
                st.info("Aucune phrase détectée pour ce discours.")
                continue

            st.dataframe(
                df_res[["id_phrase", "texte", "label", "score", "valence", "valence_lissee"]],
                use_container_width=True,
            )
            _visualiser_courbe(df_res, titre=f"Courbe de valence émotionnelle — {nom}")

