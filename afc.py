# -*- coding: utf-8 -*-
"""Onglet AFC (analyse factorielle des correspondances).

Ce module construit une table de contingence à partir des détections
réalisées sur les discours et propose une AFC simplifiée (sans
dépendance externe). Les lignes correspondent aux discours complets ou
aux phrases, et les colonnes regroupent différents types de marqueurs
ou connecteurs (normes, causes/conséquences, mémoire, tensions
semantiques, etc.).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from text_utils import normaliser_espace, segmenter_en_phrases


@dataclass
class Segment:
    """Représente un segment de discours (discours complet ou phrase)."""

    label: str
    texte: str
    id_phrase: Optional[int] = None


def _construire_segments(texte: str, label_discours: str, mode: str) -> List[Segment]:
    texte_norm = normaliser_espace(texte)
    if not texte_norm:
        return []

    if mode == "Discours complet":
        return [Segment(label=label_discours, texte=texte_norm, id_phrase=None)]

    segments = []
    for i, phrase in enumerate(segmenter_en_phrases(texte_norm), start=1):
        segments.append(
            Segment(
                label=f"{label_discours} – phrase {i}",
                texte=phrase,
                id_phrase=i,
            )
        )
    return segments


def _ajouter_comptages(
    df: pd.DataFrame,
    *,
    colonne_categorie: str,
    prefixe: str,
    segments: Sequence[Segment],
    compteurs: Dict[str, Counter],
    mode: str,
) -> None:
    if df.empty:
        return

    index_segment = {seg.id_phrase: seg.label for seg in segments if seg.id_phrase is not None}
    label_discours = segments[0].label if segments else None

    for _, row in df.iterrows():
        if mode == "Discours complet":
            label = label_discours
        else:
            label = index_segment.get(row.get("id_phrase"))
        if not label:
            continue
        categorie = str(row.get(colonne_categorie, "")).strip()
        if not categorie:
            continue
        cle = f"{prefixe}{categorie.upper()}"
        compteurs.setdefault(label, Counter())[cle] += 1


def _table_contingence(
    segments: Sequence[Segment],
    compteurs: Dict[str, Counter],
) -> pd.DataFrame:
    if not segments:
        return pd.DataFrame()

    toutes_colonnes = sorted({cle for c in compteurs.values() for cle in c})
    if not toutes_colonnes:
        return pd.DataFrame()

    lignes = []
    index = []
    for seg in segments:
        cpt = compteurs.get(seg.label, Counter())
        lignes.append([cpt.get(col, 0) for col in toutes_colonnes])
        index.append(seg.label)
    return pd.DataFrame(lignes, index=index, columns=toutes_colonnes)


def _heatmap_contingence(table: pd.DataFrame) -> alt.Chart:
    table_reset = table.reset_index().rename(columns={"index": "Segment"})
    data = table_reset.melt(id_vars="Segment", var_name="Variable", value_name="Comptage")
    return (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X("Variable", sort=None),
            y=alt.Y("Segment", sort=None),
            color=alt.Color("Comptage", scale=alt.Scale(scheme="blues")),
            tooltip=["Segment", "Variable", alt.Tooltip("Comptage", format="d")],
        )
        .properties(height=260)
    )


def _analyse_afc(table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[float]]:
    """Calcule une AFC minimale (deux premiers axes)."""

    n = table.values.sum()
    if n <= 0:
        raise ValueError("La table de contingence est vide.")

    P = table / n
    r = P.sum(axis=1)
    c = P.sum(axis=0)

    # Suppression des lignes/colonnes nulles pour éviter les divisions par zéro
    lignes_valides = r > 0
    colonnes_valides = c > 0
    P = P.loc[lignes_valides, colonnes_valides]
    r = P.sum(axis=1)
    c = P.sum(axis=0)

    R_inv = np.diag(1 / np.sqrt(r.values))
    C_inv = np.diag(1 / np.sqrt(c.values))

    deviation = P - np.outer(r, c)
    S = R_inv @ deviation.values @ C_inv
    U, singular_values, Vt = np.linalg.svd(S, full_matrices=False)

    eigvals = (singular_values**2).tolist()
    dims = min(2, len(singular_values))
    inv_sv = np.diag(1 / singular_values[:dims])

    row_coords = R_inv @ deviation.values @ C_inv @ Vt.T[:, :dims] @ inv_sv
    col_coords = C_inv @ deviation.values.T @ R_inv @ U[:, :dims] @ inv_sv

    row_df = pd.DataFrame(
        row_coords,
        index=P.index,
        columns=[f"Dim {i+1}" for i in range(dims)],
    )
    col_df = pd.DataFrame(
        col_coords,
        index=P.columns,
        columns=[f"Dim {i+1}" for i in range(dims)],
    )
    return row_df, col_df, eigvals


def _scatter_afc(row_df: pd.DataFrame, col_df: pd.DataFrame) -> alt.Chart:
    def _prep(df: pd.DataFrame, typ: str) -> pd.DataFrame:
        return df.reset_index().rename(columns={"index": "libelle"}).assign(Type=typ)

    data = pd.concat([_prep(row_df, "Segments"), _prep(col_df, "Variables")], ignore_index=True)
    tooltip = ["libelle", "Type", alt.Tooltip("Dim 1", format=".3f"), alt.Tooltip("Dim 2", format=".3f")]
    color = alt.condition(alt.datum.Type == "Segments", alt.value("#1f77b4"), alt.value("#d62728"))

    return (
        alt.Chart(data)
        .mark_circle(size=120, opacity=0.8)
        .encode(
            x="Dim 1",
            y="Dim 2",
            color=color,
            tooltip=tooltip,
        )
        .properties(height=420)
    )


def render_afc_tab(
    texte_source: str,
    texte_source_2: str,
    detections_1: Dict[str, pd.DataFrame],
    detections_2: Dict[str, pd.DataFrame],
    libelle_discours_1: str,
    libelle_discours_2: str,
) -> None:
    st.subheader("Analyse factorielle des correspondances (AFC)")
    st.caption(
        "Construisez une table de contingence à partir des marqueurs détectés, puis projetez les segments de discours et les variables dans un plan factoriel."
    )

    mode = st.radio("Granularité des lignes", ["Discours complet", "Phrases"], horizontal=True)
    st.markdown("**Variables incluses dans l'AFC**")
    col1, col2, col3 = st.columns(3)
    use_marqueurs = col1.checkbox("Marqueurs normatifs", value=True)
    use_connecteurs = col1.checkbox("Connecteurs logiques", value=True)
    use_memoires = col2.checkbox("Mémoire", value=True)
    use_causes = col2.checkbox("Causes", value=True)
    use_consqs = col3.checkbox("Conséquences", value=True)
    use_tensions = col3.checkbox("Tensions sémantiques", value=True)

    segments_1 = _construire_segments(texte_source, libelle_discours_1, mode)
    segments_2 = _construire_segments(texte_source_2, libelle_discours_2, mode)
    segments = segments_1 + segments_2

    if not segments:
        st.info("Aucun segment disponible (veuillez charger ou saisir un discours).")
        return

    compteurs: Dict[str, Counter] = {}

    if use_marqueurs:
        _ajouter_comptages(
            detections_1.get("df_marq", pd.DataFrame()),
            colonne_categorie="categorie",
            prefixe="MARQ_",
            segments=segments_1,
            compteurs=compteurs,
            mode=mode,
        )
        _ajouter_comptages(
            detections_2.get("df_marq", pd.DataFrame()),
            colonne_categorie="categorie",
            prefixe="MARQ_",
            segments=segments_2,
            compteurs=compteurs,
            mode=mode,
        )

    if use_connecteurs:
        _ajouter_comptages(
            detections_1.get("df_conn", pd.DataFrame()),
            colonne_categorie="code",
            prefixe="CONN_",
            segments=segments_1,
            compteurs=compteurs,
            mode=mode,
        )
        _ajouter_comptages(
            detections_2.get("df_conn", pd.DataFrame()),
            colonne_categorie="code",
            prefixe="CONN_",
            segments=segments_2,
            compteurs=compteurs,
            mode=mode,
        )

    if use_memoires:
        _ajouter_comptages(
            detections_1.get("df_memoires", pd.DataFrame()),
            colonne_categorie="categorie",
            prefixe="MEM_",
            segments=segments_1,
            compteurs=compteurs,
            mode=mode,
        )
        _ajouter_comptages(
            detections_2.get("df_memoires", pd.DataFrame()),
            colonne_categorie="categorie",
            prefixe="MEM_",
            segments=segments_2,
            compteurs=compteurs,
            mode=mode,
        )

    if use_causes:
        _ajouter_comptages(
            detections_1.get("df_causes_lex", pd.DataFrame()),
            colonne_categorie="categorie",
            prefixe="CAUSE_",
            segments=segments_1,
            compteurs=compteurs,
            mode=mode,
        )
        _ajouter_comptages(
            detections_2.get("df_causes_lex", pd.DataFrame()),
            colonne_categorie="categorie",
            prefixe="CAUSE_",
            segments=segments_2,
            compteurs=compteurs,
            mode=mode,
        )

    if use_consqs:
        _ajouter_comptages(
            detections_1.get("df_consq_lex", pd.DataFrame()),
            colonne_categorie="categorie",
            prefixe="CONSQ_",
            segments=segments_1,
            compteurs=compteurs,
            mode=mode,
        )
        _ajouter_comptages(
            detections_2.get("df_consq_lex", pd.DataFrame()),
            colonne_categorie="categorie",
            prefixe="CONSQ_",
            segments=segments_2,
            compteurs=compteurs,
            mode=mode,
        )

    if use_tensions:
        _ajouter_comptages(
            detections_1.get("df_tensions", pd.DataFrame()),
            colonne_categorie="tension",
            prefixe="TENS_",
            segments=segments_1,
            compteurs=compteurs,
            mode=mode,
        )
        _ajouter_comptages(
            detections_2.get("df_tensions", pd.DataFrame()),
            colonne_categorie="tension",
            prefixe="TENS_",
            segments=segments_2,
            compteurs=compteurs,
            mode=mode,
        )

    table = _table_contingence(segments, compteurs)

    if table.empty:
        st.info("Aucune variable sélectionnée ou aucune occurrence détectée dans les segments.")
        return

    st.markdown("### Table de contingence")
    st.dataframe(table, use_container_width=True)
    st.altair_chart(_heatmap_contingence(table), use_container_width=True)
    st.download_button(
        "Exporter la table (CSV)",
        data=table.to_csv().encode("utf-8"),
        file_name="table_afc.csv",
        mime="text/csv",
        key="dl_table_afc",
    )

    try:
        row_df, col_df, eigvals = _analyse_afc(table)
    except Exception as exc:  # pragma: no cover - robustesse interactive
        st.error(f"AFC impossible : {exc}")
        return

    inertie_totale = sum(eigvals)
    if inertie_totale > 0:
        inertie_dim1 = eigvals[0] / inertie_totale * 100 if eigvals else 0
        inertie_dim2 = eigvals[1] / inertie_totale * 100 if len(eigvals) > 1 else 0
    else:
        st.warning(
            "Inertie totale nulle : impossible de calculer les contributions des dimensions."
        )
        inertie_dim1 = 0
        inertie_dim2 = 0

    st.markdown(
        f"**Inertie** – Dim 1 : {inertie_dim1:.1f}% · Dim 2 : {inertie_dim2:.1f}%"
    )
    st.altair_chart(_scatter_afc(row_df, col_df), use_container_width=True)

    st.markdown("### Coordonnées factorielles")
    st.write("**Segments**")
    st.dataframe(row_df, use_container_width=True)
    st.write("**Variables**")
    st.dataframe(col_df, use_container_width=True)
