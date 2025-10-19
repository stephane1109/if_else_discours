"""Outils de statistiques et de visualisation pour les marqueurs détectés."""
from __future__ import annotations

import re
from typing import List

import pandas as pd
import streamlit as st
import altair as alt


def _segmenter_en_phrases_local(texte: str) -> List[str]:
    """Segmente approximativement le texte en phrases en s'alignant sur la logique de main.py."""
    if not texte:
        return []
    morceaux = re.split(r"(?<=[\.\!\?\:\;])\s+", texte)
    return [m.strip() for m in morceaux if m and m.strip()]


def _positions_phrases(texte: str):
    """Calcule pour chaque phrase l’offset caractère de début pour projeter les positions en % du texte."""
    phrases = _segmenter_en_phrases_local(texte)
    offsets = []
    pos = 0
    t = texte
    for ph in phrases:
        i = t.find(ph, pos)
        if i < 0:
            i = pos
        offsets.append(i)
        pos = i + len(ph)
    return phrases, offsets, len(texte)


def _ajoute_colonne_t_rel(
    df: pd.DataFrame,
    phrases: List[str],
    offsets: List[int],
    total_len: int,
    col_id: str = "id_phrase",
    col_pos: str = "position",
):
    """Ajoute une colonne t_rel ∈ [0,100], position relative du marqueur dans le texte."""
    if df.empty:
        return df
    m = df.copy()

    def t_rel_from_row(r: pd.Series) -> float:
        try:
            i = int(r[col_id]) - 1
            base = offsets[i] if 0 <= i < len(offsets) else 0
            pos_abs = base + int(r.get(col_pos, 0))
        except Exception:
            pos_abs = 0
        if total_len <= 0:
            return 0.0
        return round(100.0 * pos_abs / total_len, 3)

    m["t_rel"] = m.apply(t_rel_from_row, axis=1)
    return m


def construire_df_temps(
    texte_source: str,
    df_conn: pd.DataFrame,
    df_marq: pd.DataFrame,
    df_consq_lex: pd.DataFrame,
    df_causes_lex: pd.DataFrame,
) -> pd.DataFrame:
    """Fusionne tous les jeux en un seul tableau temporel avec colonnes normalisées pour Altair."""
    if not texte_source or not texte_source.strip():
        return pd.DataFrame()

    phrases, offsets, total_len = _positions_phrases(texte_source)

    a = df_conn.copy()
    if not a.empty:
        a["type"] = "CONNECTEUR"
        a.rename(columns={"connecteur": "surface", "code": "etiquette"}, inplace=True)
        a = _ajoute_colonne_t_rel(a, phrases, offsets, total_len)

    b = df_marq.copy()
    if not b.empty:
        b["type"] = "MARQUEUR"
        b.rename(columns={"marqueur": "surface", "categorie": "etiquette"}, inplace=True)
        b = _ajoute_colonne_t_rel(b, phrases, offsets, total_len)

    c = df_consq_lex.copy()
    if not c.empty:
        c["type"] = "CONSEQUENCE"
        c.rename(columns={"consequence": "surface", "categorie": "etiquette"}, inplace=True)
        c = _ajoute_colonne_t_rel(c, phrases, offsets, total_len)

    d = df_causes_lex.copy()
    if not d.empty:
        d["type"] = "CAUSE"
        d.rename(columns={"cause": "surface", "categorie": "etiquette"}, inplace=True)
        d = _ajoute_colonne_t_rel(d, phrases, offsets, total_len)

    frames = [x for x in [a, b, c, d] if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["etiquette"] = (
        df["etiquette"].astype(str).str.strip().replace("", pd.NA).fillna("INCONNU")
    )
    cols = [
        "t_rel",
        "id_phrase",
        "surface",
        "etiquette",
        "type",
        "position",
        "longueur",
        "phrase",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    return df[cols]


def graphique_altair_chronologie(
    df_temps: pd.DataFrame,
    filtres_types: List[str] | None = None,
    filtres_etiquettes: List[str] | None = None,
):
    """Construit un scatter Altair chronologique; filtres optionnels sur type et étiquette."""
    if df_temps.empty:
        return None

    data = df_temps.copy()
    if filtres_types:
        data = data[data["type"].isin(filtres_types)]
    if filtres_etiquettes:
        data = data[data["etiquette"].isin([e.upper() for e in filtres_etiquettes])]

    base = (
        alt.Chart(data)
        .mark_point(filled=True)
        .encode(
            x=alt.X(
                "t_rel:Q",
                title="Progression du discours (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y("etiquette:N", title="Famille / Catégorie"),
            color=alt.Color("type:N", title="Type", legend=alt.Legend(title="Type")),
            size=alt.Size("longueur:Q", title="Longueur repérée", legend=None),
            tooltip=[
                alt.Tooltip("t_rel:Q", title="Progression (%)", format=".2f"),
                alt.Tooltip("id_phrase:Q", title="Phrase #"),
                alt.Tooltip("surface:N", title="Surface"),
                alt.Tooltip("etiquette:N", title="Famille/Cat."),
                alt.Tooltip("type:N", title="Type"),
                alt.Tooltip("phrase:N", title="Phrase"),
            ],
        )
        .properties(height=320)
    )

    return base


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


def construire_tableau_marqueurs_simples(
    texte_source: str,
    df_conn: pd.DataFrame,
    df_marq: pd.DataFrame,
    df_consq_lex: pd.DataFrame,
    df_causes_lex: pd.DataFrame,
) -> pd.DataFrame:
    """Unifie connecteurs, marqueurs normatifs, causes et conséquences en un seul tableau standardisé."""
    cadres = []

    if df_conn is not None and not df_conn.empty:
        a = df_conn.copy()
        a["type"] = "CONNECTEUR"
        a.rename(columns={"connecteur": "surface", "code": "etiquette"}, inplace=True)
        cadres.append(a[["type", "etiquette", "surface"]])

    if df_marq is not None and not df_marq.empty:
        b = df_marq.copy()
        b["type"] = "MARQUEUR"
        b.rename(columns={"marqueur": "surface", "categorie": "etiquette"}, inplace=True)
        cadres.append(b[["type", "etiquette", "surface"]])

    if df_consq_lex is not None and not df_consq_lex.empty:
        c = df_consq_lex.copy()
        c["type"] = "CONSEQUENCE"
        c.rename(columns={"consequence": "surface", "categorie": "etiquette"}, inplace=True)
        cadres.append(c[["type", "etiquette", "surface"]])

    if df_causes_lex is not None and not df_causes_lex.empty:
        d = df_causes_lex.copy()
        d["type"] = "CAUSE"
        d.rename(columns={"cause": "surface", "categorie": "etiquette"}, inplace=True)
        cadres.append(d[["type", "etiquette", "surface"]])

    if not cadres:
        return pd.DataFrame(columns=["type", "etiquette", "surface"])

    df = pd.concat(cadres, ignore_index=True)
    return df


def graphique_frequences_barres(df_unifie: pd.DataFrame, niveau: str = "etiquette"):
    """Affiche un histogramme de fréquences agrégées par étiquette ou type."""

    if df_unifie is None or df_unifie.empty:
        return None

    champ = "etiquette" if niveau == "etiquette" else "type"
    agg = df_unifie.groupby([champ], dropna=False).size().reset_index(name="freq")

    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("freq:Q", title="Fréquence"),
            y=alt.Y(f"{champ}:N", sort="-x", title="Catégorie"),
            color=alt.Color(f"{champ}:N", legend=None),
            tooltip=[
                alt.Tooltip(f"{champ}:N", title="Catégorie"),
                alt.Tooltip("freq:Q", title="Fréquence"),
            ],
        )
        .properties(height=320)
    )
    return chart


def render_stats_tab(
    texte_source: str,
    df_conn: pd.DataFrame,
    df_marqueurs: pd.DataFrame,
    df_consq_lex: pd.DataFrame,
    df_causes_lex: pd.DataFrame,
) -> None:
    """Affiche les statistiques des marqueurs dans l'onglet Streamlit dédié."""
    st.subheader("Statistiques des marqueurs normatifs")

    if df_marqueurs is None or df_marqueurs.empty:
        st.info("Aucun marqueur détecté pour générer des statistiques.")
        df = pd.DataFrame()
    else:
        df = df_marqueurs.copy()
        df["categorie"] = df["categorie"].astype(str).str.upper()
        df["marqueur"] = df["marqueur"].astype(str)

        total_occurrences = len(df)
        total_categories = df["categorie"].nunique(dropna=True)
        total_marqueurs_distincts = df["marqueur"].nunique(dropna=True)
        total_phrases = (
            df["id_phrase"].nunique(dropna=True) if "id_phrase" in df.columns else None
        )

        cols = st.columns(4 if total_phrases is not None else 3)
        cols[0].metric("Occurrences", f"{total_occurrences}")
        cols[1].metric("Catégories distinctes", f"{total_categories}")
        cols[2].metric("Marqueurs distincts", f"{total_marqueurs_distincts}")
        if total_phrases is not None:
            cols[3].metric("Phrases concernées", f"{total_phrases}")

        st.markdown("---")

        st.markdown("### Fréquences des marqueurs (histogramme)")
        df_unifie_simple = construire_tableau_marqueurs_simples(
            texte_source,
            df_conn,
            df_marqueurs,
            df_consq_lex,
            df_causes_lex,
        )

        if df_unifie_simple.empty:
            st.info("Aucune donnée détectée pour calculer les fréquences.")
        else:
            choix_niveau = st.radio(
                "Regrouper par",
                ["Familles (étiquette)", "Macro-types (type)"],
                horizontal=True,
                index=0,
            )
            niveau = "etiquette" if "Familles" in choix_niveau else "type"
            chart = graphique_frequences_barres(df_unifie_simple, niveau=niveau)
            if chart is None:
                st.info("Rien à afficher avec les filtres actuels.")
            else:
                st.altair_chart(chart, use_container_width=True)

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

        st.markdown("---")

    if df.empty:
        st.markdown("---")

    st.markdown("### Chronologie des marqueurs")
    df_temps = construire_df_temps(
        texte_source=texte_source,
        df_conn=df_conn,
        df_marq=df_marqueurs,
        df_consq_lex=df_consq_lex,
        df_causes_lex=df_causes_lex,
    )

    if df_temps.empty:
        options_familles: List[str] = []
    else:
        etiquettes = (
            df_temps["etiquette"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
        )
        options_familles = sorted(str(e) for e in etiquettes)

    choix_familles = st.multiselect(
        "Filtrer par famille/catégorie",
        options_familles,
        default=[],
    )

    if df_temps.empty:
        st.info("Aucune détection pour construire la chronologie.")
    else:
        chart = graphique_altair_chronologie(
            df_temps,
            filtres_etiquettes=choix_familles,
        )
        if chart is None:
            st.info("Rien à afficher avec les filtres actuels.")
        else:
            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "Chaque point représente une occurrence détectée, positionnée en pourcentage du texte."
            )

