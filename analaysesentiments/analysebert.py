"""Onglet d'analyse CamemBERT (classification de sentiments) pour l'application Streamlit."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
import streamlit as st
import altair as alt
from transformers import AutoTokenizer, CamembertTokenizer, pipeline

from text_utils import normaliser_espace, segmenter_en_phrases


STAR_TO_VALENCE = {
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}

VALEUR_BADGES = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}


@st.cache_resource(show_spinner=False)
def _charger_camembert_pipeline():
    """Charge la pipeline CamemBERT pour la classification de sentiments."""

    try:
        try:
            tokenizer = CamembertTokenizer.from_pretrained(
                "cmarkea/distilcamembert-base-sentiment"
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                "cmarkea/distilcamembert-base-sentiment", use_fast=False
            )
        return pipeline(
            "text-classification",
            model="cmarkea/distilcamembert-base-sentiment",
            tokenizer=tokenizer,
        )
    except Exception as exc:  # pragma: no cover - uniquement déclenché en environnement Streamlit
        st.error(
            "Impossible de charger CamemBERT (sentiment). Vérifiez la connexion et les dépendances nécessaires."
        )
        st.exception(exc)
        return None


def _construire_df_sentiments(phrases: List[str], predictions) -> pd.DataFrame:
    """Convertit les scores du modèle en DataFrame lisible par phrase."""

    if not predictions:
        return pd.DataFrame(
            columns=["id_phrase", "texte_phrase", "valence", "score_valence"]
        )

    lignes = []
    for idx, (phrase, scores) in enumerate(zip(phrases, predictions), start=1):
        if not scores:
            continue

        scores_valence = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        for score in scores:
            etiquette = score.get("label", "")
            valence = STAR_TO_VALENCE.get(etiquette.lower())
            if valence:
                scores_valence[valence] += score.get("score", 0)

        meilleure_valence = max(scores_valence, key=scores_valence.get)
        ligne = {
            "id_phrase": idx,
            "texte_phrase": phrase,
            "valence": meilleure_valence,
            "score_valence": scores_valence[meilleure_valence],
        }

        for nom_valence, val in scores_valence.items():
            ligne[f"score_{nom_valence}"] = val

        for score in scores:
            etiquette = score.get("label", "").lower().replace(" ", "_")
            ligne[f"score_{etiquette}"] = score.get("score", 0)

        lignes.append(ligne)

    return pd.DataFrame(lignes)


def _tracer_barres_scores(df_sentiments: pd.DataFrame):
    """Affiche un graphique Altair des scores par phrase."""

    if df_sentiments.empty:
        st.info("Aucune phrase à représenter.")
        return

    df_barres = df_sentiments.copy()
    chart = (
        alt.Chart(df_barres)
        .mark_bar()
        .encode(
            x=alt.X("id_phrase:O", title="Phrase"),
            y=alt.Y("score_valence:Q", title="Score agrégé (valence)"),
            color=alt.Color(
                "valence:N",
                title="Valence",
                scale=alt.Scale(
                    domain=["positive", "neutral", "negative"],
                    range=["seagreen", "lightgray", "indianred"],
                ),
            ),
            tooltip=[
                "id_phrase",
                "valence",
                alt.Tooltip("score_valence:Q", format=".3f"),
            ],
        )
        .properties(height=300, width="container")
    )
    st.altair_chart(chart, use_container_width=True)


def _tracer_moyennes(df_sentiments: pd.DataFrame):
    """Affiche un graphique des moyennes des scores par sentiment."""

    colonnes_scores = [
        col for col in df_sentiments.columns if col in {"score_positive", "score_neutral", "score_negative"}
    ]
    if not colonnes_scores:
        return

    df_moyennes = (
        df_sentiments[colonnes_scores]
        .mean()
        .reset_index()
        .rename(columns={"index": "sentiment", 0: "score"})
    )
    df_moyennes["sentiment"] = df_moyennes["sentiment"].str.replace("score_", "")

    chart = (
        alt.Chart(df_moyennes)
        .mark_bar()
        .encode(
            x=alt.X("sentiment:N", title="Sentiment"),
            y=alt.Y("score:Q", title="Score moyen agrégé"),
            color=alt.Color(
                "sentiment:N",
                scale=alt.Scale(
                    domain=["positive", "neutral", "negative"],
                    range=["seagreen", "lightgray", "indianred"],
                ),
            ),
            tooltip=["sentiment", alt.Tooltip("score:Q", format=".3f")],
        )
        .properties(height=250, width="container")
    )
    st.altair_chart(chart, use_container_width=True)


def _selectionner_texte(
    texte_discours_1: str, texte_discours_2: str, nom_discours_1: str, nom_discours_2: str
) -> str:
    """Offre une sélection rapide entre les deux discours et une zone d'édition."""

    textes_disponibles: Dict[str, str] = {}
    if texte_discours_1.strip():
        textes_disponibles[nom_discours_1] = texte_discours_1
    if texte_discours_2.strip():
        textes_disponibles[nom_discours_2] = texte_discours_2

    choix = None
    if textes_disponibles:
        choix = st.selectbox(
            "Choisissez un discours à charger dans la zone de test",
            options=list(textes_disponibles.keys()),
            help="Le texte sélectionné est pré-rempli ci-dessous pour l'inférence CamemBERT.",
        )

    contenu_initial = textes_disponibles.get(choix, "") if choix else ""
    return st.text_area(
        "Texte à analyser",
        value=contenu_initial
        or "C'est formidable de voir tout le monde aujourd'hui pour échanger ensemble !",
        height=200,
    )


def render_camembert_tab(
    texte_discours_1: str, texte_discours_2: str, nom_discours_1: str, nom_discours_2: str
):
    """Rendu Streamlit pour l'onglet AnalysSentCamemBert."""

    st.markdown("### AnalysSentCamemBert")
    st.caption(
        "Analyse de sentiments en français basée sur cmarkea/distilcamembert-base-sentiment"
        " (étiquettes étoiles regroupées en valence positive / neutre / négative)."
    )

    st.markdown(
        """
        Phrase exemple : "Mesdames et messieurs les parlementaires, il faut savoir tirer les bienfaits d'une crise."\
        Approche par intelligence artificielle (CamemBERT) contre l'approche "dictionnaire" (VADER).\
        Différence d'interprétation :
        * VADER (Dictionnaire) : le mot "crise" ➡️ Négatif.
        * CamemBERT (Contexte) : A lu la phrase entière ("tirer les bienfaits") ➡️ Positif (0.78).
        """
    )

    st.markdown(
        """
        **Comment fonctionne cette analyse ?**

        * Le modèle [CamemBERT](https://huggingface.co/cmarkea/distilcamembert-base-sentiment) est spécialisé pour le français.
        * Chaque discours est découpé en phrases avant d'être envoyé au modèle de classification.
        * Les étiquettes « 1 à 5 étoiles » sont converties en trois sentiments (positif, neutre, négatif) puis agrégées pour donner un score par phrase.
        * Les tableaux et graphiques ci-dessous affichent ces scores pour visualiser la polarité générale du texte.
        """
    )

    texte_cible = _selectionner_texte(texte_discours_1, texte_discours_2, nom_discours_1, nom_discours_2)
    texte_cible = normaliser_espace(texte_cible)

    seuil_affichage = st.slider(
        "Seuil minimal de probabilité (valence agrégée)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        help="Les phrases dont le score agrégé est inférieur à ce seuil sont masquées dans les résultats.",
    )
    st.caption(
        "Plus vous augmentez ce seuil, plus seules les phrases dont la valence est clairement"
        " positive ou négative resteront affichées ; un seuil bas laisse passer les phrases"
        " à tonalité plus nuancée."
    )

    if "camembert_pipe" not in st.session_state:
        st.session_state["camembert_pipe"] = None

    if st.button("Lancer l'import CamemBERT", type="primary"):
        with st.spinner("Import et initialisation du modèle CamemBERT..."):
            st.session_state["camembert_pipe"] = _charger_camembert_pipeline()

        if st.session_state["camembert_pipe"] is None:
            st.warning(
                "Le modèle n'a pas pu être chargé. Vérifiez les dépendances puis réessayez."
            )
        else:
            st.success("CamemBERT est prêt pour l'analyse de sentiments.")

    if st.session_state.get("camembert_pipe") is None:
        st.info(
            "Cliquez sur le bouton ci-dessus pour importer et initialiser CamemBERT avant l'analyse."
        )
        return

    if st.button("Lancer l'analyse CamemBERT"):
        with st.spinner("Inférence en cours..."):
            if st.session_state.get("camembert_pipe") is None:
                st.warning(
                    "Le modèle CamemBERT n'a pas été initialisé. Cliquez d'abord sur le bouton d'import."
                )
                return

            if not texte_cible:
                st.warning("Veuillez saisir un texte avant de lancer l'analyse.")
                return

            phrases = segmenter_en_phrases(texte_cible) or [texte_cible]
            predictions = st.session_state["camembert_pipe"](
                phrases, return_all_scores=True
            )
            df_sentiments = _construire_df_sentiments(phrases, predictions)
            df_affiches = df_sentiments[df_sentiments["score_valence"] >= seuil_affichage]

        st.success("Analyse CamemBERT terminée.")
        if df_affiches.empty:
            st.info("Aucun résultat atteint le seuil de probabilité sélectionné.")
            return

        if len(df_affiches) < len(df_sentiments):
            st.caption(
                f"{len(df_affiches)} phrase(s) affichée(s) sur {len(df_sentiments)} après application du seuil."
            )

        st.markdown("#### Texte annoté")
        for ligne in df_affiches.itertuples():
            badge = VALEUR_BADGES.get(ligne.valence.lower(), "🔎")
            st.markdown(
                f"{badge} **Phrase {ligne.id_phrase}** — {ligne.valence}"
                f" (score {ligne.score_valence:.3f}) : {ligne.texte_phrase}"
            )

        st.markdown("#### Tableau des scores")
        st.dataframe(df_affiches, use_container_width=True)

        st.markdown("#### Graphiques des sentiments")
        _tracer_barres_scores(df_affiches)
        _tracer_moyennes(df_affiches)
