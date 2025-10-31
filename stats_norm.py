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


def render_stats_norm_tab(texte_source: str) -> None:
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

    st.markdown("---")

    df = _fabrique_tableau(
        [
            RatioResult("Occurrences de « si »", nb_si, ratio_si),
            RatioResult("Schémas ‘si … alors’", nb_schemas, ratio_schema),
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(
        "Les ratios sont calculés en divisant le nombre d’occurrences par le volume total de mots, "
        "puis multipliés par 1 000 pour neutraliser l’effet de longueur du discours."
    )
