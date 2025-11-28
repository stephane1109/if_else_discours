"""Onglet d'analyse CamemBERT (fill-mask) pour l'application Streamlit."""
from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st
from transformers import pipeline

from text_utils import normaliser_espace


@st.cache_resource(show_spinner=False)
def _charger_camembert_pipeline():
    """Charge la pipeline CamemBERT pour l'inférence fill-mask."""

    return pipeline("fill-mask", model="cmarkea/distilcamembert-base")


def _precharger_si_demande():
    """Précharge CamemBERT si l'utilisateur a activé le chargement automatique."""

    if st.session_state.get("camembert_autoload") and not st.session_state.get(
        "camembert_charge"
    ):
        with st.spinner(
            "Chargement automatique du modèle CamemBERT activé, merci de patienter..."
        ):
            _charger_camembert_pipeline()
        st.session_state["camembert_charge"] = True
        st.success("CamemBERT a été téléchargé et initialisé automatiquement.")


def _construire_df_predictions(predictions) -> pd.DataFrame:
    """Convertit les prédictions de CamemBERT en DataFrame lisible."""

    if not predictions:
        return pd.DataFrame(columns=["token_str", "score", "sequence"])
    return pd.DataFrame(predictions)[["token_str", "score", "sequence"]]


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
        "Texte (inclure un '<mask>' pour la prédiction)",
        value=contenu_initial or "C'est <mask> de voir tout le monde aujourd'hui.",
        height=200,
    )


def render_camembert_tab(
    texte_discours_1: str, texte_discours_2: str, nom_discours_1: str, nom_discours_2: str
):
    """Rendu Streamlit pour l'onglet AnalysSentCamemBert."""

    st.markdown("### AnalysSentCamemBert")
    st.caption(
        "Télécharge et active CamemBERT (fill-mask) pour explorer rapidement les sentiments ou compléter des phrases."
    )

    texte_cible = _selectionner_texte(texte_discours_1, texte_discours_2, nom_discours_1, nom_discours_2)
    texte_cible = normaliser_espace(texte_cible)

    if "camembert_charge" not in st.session_state:
        st.session_state["camembert_charge"] = False
    if "camembert_autoload" not in st.session_state:
        st.session_state["camembert_autoload"] = False

    _precharger_si_demande()

    col_charger, col_inferer = st.columns([1, 2])
    with col_charger:
        if st.button("Télécharger / Charger CamemBERT", type="primary"):
            with st.spinner("Téléchargement et initialisation du modèle CamemBERT..."):
                _charger_camembert_pipeline()
            st.session_state["camembert_charge"] = True
            st.success("CamemBERT est prêt pour l'inférence.")

        st.checkbox(
            "Charger automatiquement CamemBERT à l'ouverture de l'onglet",
            key="camembert_autoload",
            help=(
                "Permet de précharger le modèle dès l'accès à l'onglet pour limiter les erreurs "
                "lors du lancement de l'application et éviter de charger plusieurs modèles simultanément."
            ),
        )

    if not st.session_state.get("camembert_charge"):
        st.info("Cliquez sur le bouton ci-dessus pour télécharger et initialiser CamemBERT avant l'analyse.")
        return

    if "<mask>" not in texte_cible:
        st.warning("Le texte doit contenir un jeton '<mask>' pour lancer la prédiction fill-mask.")
        return

    with col_inferer:
        if st.button("Lancer l'analyse CamemBERT"):
            with st.spinner("Inférence en cours..."):
                pipe = _charger_camembert_pipeline()
                predictions = pipe(texte_cible)
                df_predictions = _construire_df_predictions(predictions)
            st.success("Analyse CamemBERT terminée.")
            if df_predictions.empty:
                st.info("Aucune prédiction disponible.")
            else:
                st.dataframe(df_predictions, use_container_width=True)
