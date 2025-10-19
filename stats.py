"""Outils de statistiques et de visualisation pour les marqueurs détectés."""
from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st
import altair as alt


def _value_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Retourne les occurrences d'une colonne en normalisant les valeurs texte."""
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "occurrences"])

    series = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )
    if series.empty:
        return pd.DataFrame(columns=[column, "occurrences"])

    counts = (
        series.str.upper()
        .value_counts()
        .rename_axis(column)
        .reset_index(name="occurrences")
        .sort_values("occurrences", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    return counts


def _marqueurs_par_categorie(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Construit un dictionnaire categorie -> DataFrame des marqueurs classés par fréquence."""
    if df.empty or "categorie" not in df.columns or "marqueur" not in df.columns:
        return {}

    tmp = df[["categorie", "marqueur"]].copy()
    tmp["categorie"] = tmp["categorie"].astype(str).str.upper()
    tmp["marqueur"] = tmp["marqueur"].astype(str)

    grouped = (
        tmp.groupby(["categorie", "marqueur"], dropna=False)
        .size()
        .rename("occurrences")
        .reset_index()
    )

    resultat: Dict[str, pd.DataFrame] = {}
    for categorie, bloc in grouped.groupby("categorie"):
        tri = bloc.sort_values("occurrences", ascending=False, kind="mergesort")
        resultat[categorie] = tri.reset_index(drop=True)
    return resultat


def render_stats_tab(df_marqueurs: pd.DataFrame) -> None:
    """Affiche les statistiques des marqueurs dans l'onglet Streamlit dédié."""
    st.subheader("Statistiques des marqueurs normatifs")

    if df_marqueurs is None or df_marqueurs.empty:
        st.info("Aucun marqueur détecté pour générer des statistiques.")
        return

    df = df_marqueurs.copy()
    df["categorie"] = df["categorie"].astype(str).str.upper()
    df["marqueur"] = df["marqueur"].astype(str)

    total_occurrences = len(df)
    total_categories = df["categorie"].nunique(dropna=True)
    total_marqueurs_distincts = df["marqueur"].nunique(dropna=True)
    total_phrases = df["id_phrase"].nunique(dropna=True) if "id_phrase" in df.columns else None

    cols = st.columns(4 if total_phrases is not None else 3)
    cols[0].metric("Occurrences", f"{total_occurrences}")
    cols[1].metric("Catégories distinctes", f"{total_categories}")
    cols[2].metric("Marqueurs distincts", f"{total_marqueurs_distincts}")
    if total_phrases is not None:
        cols[3].metric("Phrases concernées", f"{total_phrases}")

    st.markdown("---")

    st.markdown("### Répartition par catégorie")
    counts_cat = _value_counts(df, "categorie")
    if counts_cat.empty:
        st.info("Impossible de calculer la répartition par catégorie.")
    else:
        st.dataframe(counts_cat, use_container_width=True, hide_index=True)
        chart_data = counts_cat.set_index("categorie")
        st.bar_chart(chart_data)

    st.markdown("### Fréquence de tous les marqueurs")
    counts_marqueurs = _value_counts(df, "marqueur")
    if counts_marqueurs.empty:
        st.info("Impossible de calculer la fréquence des marqueurs.")
    else:
        st.dataframe(counts_marqueurs, use_container_width=True, hide_index=True)
        hauteur = max(200, 22 * len(counts_marqueurs))
        chart_marqueurs = (
            alt.Chart(counts_marqueurs)
            .mark_bar()
            .encode(
                y=alt.Y("marqueur:N", sort="-x", title="Marqueur"),
                x=alt.X("occurrences:Q", title="Occurrences"),
                tooltip=[
                    alt.Tooltip("marqueur:N", title="Marqueur"),
                    alt.Tooltip("occurrences:Q", title="Occurrences"),
                ],
            )
            .properties(height=hauteur)
        )
        st.altair_chart(chart_marqueurs, use_container_width=True)

    st.markdown("### Marqueurs les plus fréquents par catégorie")
    repartition = _marqueurs_par_categorie(df)
    if not repartition:
        st.info("Aucune donnée exploitable pour afficher les marqueurs par catégorie.")
    else:
        for categorie in sorted(repartition.keys()):
            st.markdown(f"**{categorie.replace('_', ' ')}**")
            top = repartition[categorie].head(10)
            st.dataframe(top, use_container_width=True, hide_index=True)

    st.markdown("---")

    if "id_phrase" in df.columns:
        st.markdown("### Chronologie de la fréquence d'apparition")
        intensite = (
            df.groupby("id_phrase")
            .size()
            .rename("occurrences")
            .reset_index()
            .sort_values("occurrences", ascending=False, kind="mergesort")
        )
        st.dataframe(intensite, use_container_width=True, hide_index=True)

        chronologie = (
            intensite.sort_values("id_phrase", ascending=True, kind="mergesort")
            .reset_index(drop=True)
        )
        chronologie["occurrences_cumulees"] = chronologie["occurrences"].cumsum()

        ligne = (
            alt.Chart(chronologie)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "id_phrase:Q",
                    title="Phrase (ordre dans le texte)",
                ),
                y=alt.Y(
                    "occurrences:Q",
                    title="Occurrences par phrase",
                ),
                tooltip=[
                    alt.Tooltip("id_phrase:Q", title="Phrase"),
                    alt.Tooltip("occurrences:Q", title="Occurrences"),
                    alt.Tooltip(
                        "occurrences_cumulees:Q",
                        title="Occurrences cumulées",
                        format=",")
                ],
            )
        )

        zone = (
            alt.Chart(chronologie)
            .mark_area(opacity=0.25)
            .encode(
                x=alt.X("id_phrase:Q", title="Phrase (ordre dans le texte)"),
                y=alt.Y("occurrences:Q", title="Occurrences par phrase"),
            )
        )

        chronologie_chart = alt.layer(zone, ligne).properties(height=320).interactive()
        st.altair_chart(chronologie_chart, use_container_width=True)

