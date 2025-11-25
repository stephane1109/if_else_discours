"""Helpers de compatibilité pour Streamlit."""

from __future__ import annotations

import inspect
import re
from typing import Any, Optional

import altair as alt
from vl_convert import vegalite_to_png

import streamlit as st


def dataframe_safe(data: Any, **kwargs):
    """Affiche un dataframe en ignorant les arguments non pris en charge.

    Certains déploiements peuvent utiliser une version de Streamlit ne
    comprenant pas encore le paramètre ``hide_index``. Cette fonction retire
    l'argument si nécessaire et réessaie en cas d'erreur de type.
    """

    params = inspect.signature(st.dataframe).parameters
    if "hide_index" not in params:
        kwargs.pop("hide_index", None)

    try:
        return st.dataframe(data, **kwargs)
    except TypeError:
        kwargs.pop("hide_index", None)
        return st.dataframe(data, **kwargs)


def slugify_filename_component(texte: str, fallback: str = "graphique") -> str:
    """Nettoie une étiquette pour l'utiliser dans un nom de fichier.

    Le rendu remplace les séquences de caractères non alphanumériques par
    des underscores et supprime les séparateurs en début/fin. Un fallback est
    utilisé si le résultat est vide.
    """

    if not texte:
        return fallback

    slug = re.sub(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ._-]+", "_", texte.strip())
    slug = slug.strip("._-")
    return slug or fallback


def chart_to_png_bytes(
    chart: Optional[alt.Chart], *, width: int = 1100, scale: float = 2.5
) -> Optional[bytes]:
    """Convertit un graphique Altair en PNG haute définition.

    Un width explicite est appliqué pour garantir une exportation lisible,
    tandis qu'un facteur d'échelle améliore la définition. Retourne ``None``
    si le rendu échoue ou si aucun graphique n'est fourni.
    """

    if chart is None:
        return None

    try:
        export_chart = chart.properties(width=width)
        return vegalite_to_png(export_chart.to_dict(), scale=scale)
    except Exception:
        return None
