"""Outils de statistiques normalisées pour les occurrences de 'si' et des schémas 'si … alors'."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import streamlit as st

_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")
_SI_PATTERN = re.compile(r"(?<![A-Za-zÀ-ÖØ-öø-ÿ'])si(?![A-Za-zÀ-ÖØ-öø-ÿ'])", flags=re.IGNORECASE)
_SI_ALORS_PATTERN = re.compile(r"\bsi\b[\s\S]{0,400}?\balors\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class RatioResult:
    """Représente un comptage et son équivalent normalisé pour 1 000 mots."""

    label: str
    total: int
    pour_mille: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "Indicateur": self.label,
            "Total": self.total,
            "Pour 1 000 mots": round(self.pour_mille, 2),
        }


def _compter_mots(texte: str) -> int:
    """Compte grossièrement les mots en se basant sur les caractères alphabétiques."""
    if not texte:
        return 0
    return len(_WORD_PATTERN.findall(texte))


def _compter_si(texte: str) -> int:
    """Compte les occurrences isolées de « si » (insensible à la casse)."""
    if not texte:
        return 0
    return len(list(_SI_PATTERN.finditer(texte)))


def _compter_schemas_si_alors(texte: str) -> int:
    """Compte les séquences "si … alors" dans une fenêtre raisonnable."""
    if not texte:
        return 0
    return len(list(_SI_ALORS_PATTERN.finditer(texte)))


def _normaliser_par_mille(nombre: int, total_mots: int) -> float:
    """Retourne la valeur ramenée à 1 000 mots."""
    if total_mots <= 0:
        return 0.0
    return (nombre * 1000.0) / total_mots


def _fabrique_tableau(resultats: list[RatioResult]) -> pd.DataFrame:
    """Construit un DataFrame tabulaire prêt pour l'affichage."""
    if not resultats:
        return pd.DataFrame(columns=["Indicateur", "Total", "Pour 1 000 mots"])
    data = [res.to_dict() for res in resultats]
    df = pd.DataFrame(data)
    df["Pour 1 000 mots"] = df["Pour 1 000 mots"].astype(float).map(lambda v: round(v, 2))
    return df


def _series_depuis_df(df: pd.DataFrame | None, colonne: str) -> pd.Series:
    """Retourne une série nettoyée (chaînes en majuscules) pour la colonne ciblée."""

    if df is None or df.empty or colonne not in df.columns:
        return pd.Series(dtype="string")

    serie = (
        df[colonne]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )

    if serie.empty:
        return pd.Series(dtype="string")

    return serie.str.upper()


def _resultats_par_categorie(
    serie: pd.Series,
    total_mots: int,
    prefixe_label: str,
    ajouter_total: bool = True,
    format_categorie=None,
) -> list[RatioResult]:
    """Transforme une série en liste de résultats normalisés pour 1 000 mots."""

    if serie is None or serie.empty:
        return []

    compte = serie.value_counts().sort_index()
    resultats: list[RatioResult] = []

    total_occ = int(compte.sum())
    if ajouter_total:
        resultats.append(
            RatioResult(
                f"{prefixe_label} — Total",
                total_occ,
                _normaliser_par_mille(total_occ, total_mots),
            )
        )

    for categorie, nb in compte.items():
        libelle_cat = categorie
        if format_categorie is not None:
            libelle_cat = format_categorie(categorie)
        resultats.append(
            RatioResult(
                f"{prefixe_label} — {libelle_cat}",
                int(nb),
                _normaliser_par_mille(int(nb), total_mots),
            )
        )

    return resultats


def render_stats_norm_tab(
    texte_source: str,
    df_conn: pd.DataFrame,
    df_marqueurs: pd.DataFrame,
    df_memoires: pd.DataFrame,
    df_consq_lex: pd.DataFrame,
    df_causes_lex: pd.DataFrame,
) -> None:
    """Affiche les statistiques normalisées dans l'onglet dédié Streamlit."""

    st.subheader("Statistiques normalisées (par 1 000 mots)")

    if not texte_source or not texte_source.strip():
        st.info("Fournissez un texte pour calculer les ratios normalisés.")
        return

    total_mots = _compter_mots(texte_source)
    nb_si = _compter_si(texte_source)
    nb_schemas = _compter_schemas_si_alors(texte_source)

    ratio_si = _normaliser_par_mille(nb_si, total_mots)
    ratio_schema = _normaliser_par_mille(nb_schemas, total_mots)

    serie_connecteurs = _series_depuis_df(df_conn, "code")
    serie_alternatives = serie_connecteurs[serie_connecteurs == "ALTERNATIVE"]
    serie_normatifs = _series_depuis_df(df_marqueurs, "categorie")
    serie_memoires = _series_depuis_df(df_memoires, "categorie")
    serie_consq = _series_depuis_df(df_consq_lex, "categorie")
    serie_causes = _series_depuis_df(df_causes_lex, "categorie")

    col1, col2 = st.columns(2)
    col1.metric("Longueur du texte (mots)", f"{total_mots}")
    col2.metric("Référence", "1 000 mots")

    col3, col4 = st.columns(2)
    col3.metric(
        "Occurrences de « si »",
        f"{nb_si}",
        delta=f"{ratio_si:.2f} / 1 000 mots",
    )
    col4.metric(
        "Schémas ‘si … alors’",
        f"{nb_schemas}",
        delta=f"{ratio_schema:.2f} / 1 000 mots",
    )

    if not serie_alternatives.empty:
        total_alt = len(serie_alternatives)
        ratio_alt = _normaliser_par_mille(total_alt, total_mots)
        st.metric(
            "Connecteurs « sinon » / alternatifs",
            f"{total_alt}",
            delta=f"{ratio_alt:.2f} / 1 000 mots",
        )

    st.markdown("---")

    def pretty_label(categorie: str) -> str:
        return categorie.replace("_", " ")

    sections: list[tuple[str, list[RatioResult]]] = []

    sections.append(
        (
            "Repères conditionnels",
            [
                RatioResult("Occurrences de « si »", nb_si, ratio_si),
                RatioResult("Schémas ‘si … alors’", nb_schemas, ratio_schema),
            ],
        )
    )

    sections.append(
        (
            "Connecteurs",
            _resultats_par_categorie(
                serie_connecteurs,
                total_mots,
                "Connecteurs",
                ajouter_total=True,
                format_categorie=pretty_label,
            ),
        )
    )

    sections.append(
        (
            "Marqueurs normatifs",
            _resultats_par_categorie(
                serie_normatifs,
                total_mots,
                "Normatifs",
                ajouter_total=True,
                format_categorie=pretty_label,
            ),
        )
    )

    sections.append(
        (
            "Marqueurs mémoire",
            _resultats_par_categorie(
                serie_memoires,
                total_mots,
                "Mémoire",
                ajouter_total=True,
                format_categorie=pretty_label,
            ),
        )
    )

    sections.append(
        (
            "Déclencheurs de conséquence",
            _resultats_par_categorie(
                serie_consq,
                total_mots,
                "Conséquence",
                ajouter_total=True,
                format_categorie=pretty_label,
            ),
        )
    )

    sections.append(
        (
            "Déclencheurs de cause",
            _resultats_par_categorie(
                serie_causes,
                total_mots,
                "Cause",
                ajouter_total=True,
                format_categorie=pretty_label,
            ),
        )
    )

    for titre, resultats in sections:
        st.markdown(f"#### {titre}")
        if not resultats:
            st.info("Aucune donnée disponible pour cette section.")
            continue

        df_section = _fabrique_tableau(resultats)
        st.dataframe(df_section, use_container_width=True, hide_index=True)

    st.caption(
        "Les ratios sont calculés en divisant le nombre d’occurrences par le volume total de mots, "
        "puis multipliés par 1 000 pour neutraliser l’effet de longueur du discours."
    )
