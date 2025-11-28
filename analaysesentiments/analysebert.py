"""Onglet d'analyse CamemBERT (fill-mask) pour l'application Streamlit."""
from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, pipeline

from text_utils import normaliser_espace


@st.cache_resource(show_spinner=False)
def _charger_camembert_pipeline():
    """Charge la pipeline CamemBERT pour l'inférence fill-mask."""

    try:
        # Le tokenizer fast de CamemBERT peut échouer avec certaines versions de
        # `tokenizers` (vocabulaire manquant entraînant un `NoneType.endswith`).
        # On force donc l'usage du tokenizer « slow », plus robuste ici.
        tokenizer = AutoTokenizer.from_pretrained(
            "cmarkea/distilcamembert-base", use_fast=False
        )
        return pipeline(
            "fill-mask",
            model="cmarkea/distilcamembert-base",
            tokenizer=tokenizer,
        )
    except Exception as exc:  # pragma: no cover - uniquement déclenché en environnement Streamlit
        st.error(
            "Impossible de charger CamemBERT (fill-mask). Vérifiez la connexion et les dépendances nécessaires."
        )
        st.exception(exc)
        return None


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

    if "camembert_pipe" not in st.session_state:
        st.session_state["camembert_pipe"] = None

    col_charger, col_inferer = st.columns([1, 2])
    with col_charger:
        if st.button("Lancer l'import CamemBERT", type="primary"):
            with st.spinner("Import et initialisation du modèle CamemBERT..."):
                st.session_state["camembert_pipe"] = _charger_camembert_pipeline()

            if st.session_state["camembert_pipe"] is None:
                st.warning(
                    "Le modèle n'a pas pu être chargé. Vérifiez les dépendances puis réessayez."
                )
            else:
                st.success("CamemBERT est prêt pour l'inférence.")

    if st.session_state.get("camembert_pipe") is None:
        st.info(
            "Cliquez sur le bouton ci-dessus pour importer et initialiser CamemBERT avant l'analyse."
        )
        return

    if "<mask>" not in texte_cible:
        st.warning("Le texte doit contenir un jeton '<mask>' pour lancer la prédiction fill-mask.")
        return

    with col_inferer:
        if st.button("Lancer l'analyse CamemBERT"):
            with st.spinner("Inférence en cours..."):
                if st.session_state.get("camembert_pipe") is None:
                    st.warning(
                        "Le modèle CamemBERT n'a pas été initialisé. Cliquez d'abord sur le bouton d'import."
                    )
                    return

                predictions = st.session_state["camembert_pipe"](texte_cible)
                df_predictions = _construire_df_predictions(predictions)
            st.success("Analyse CamemBERT terminée.")
            if df_predictions.empty:
                st.info("Aucune prédiction disponible.")
            else:
                st.dataframe(df_predictions, use_container_width=True)
