"""Analyse de co-occurrences à partir du texte fourni.

Ce module propose un rendu Streamlit avec tableau, graphique Altair et nuage de mots
pour visualiser les co-occurrences calculées à l'échelle de la phrase ou du document.
"""
from __future__ import annotations

import math
import re
import html
from collections import Counter
from functools import lru_cache
from itertools import combinations
from typing import Iterable, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

try:  # pragma: no cover - dépendance optionnelle à l'import
    import spacy
    from spacy.language import Language
except ImportError:  # pragma: no cover - spaCy non installé
    spacy = None
    Language = None

try:  # pragma: no cover - spaCy non installé
    from spacy.lang.fr.stop_words import STOP_WORDS as SPACY_STOP_WORDS
except ImportError:  # pragma: no cover - spaCy non installé
    SPACY_STOP_WORDS = set()


_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")


_SPACY_MODELE_NOM: Optional[str] = None
_SPACY_MODELES_TENTES: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def _charger_modele_spacy() -> Optional["Language"]:
    """Charge un modèle spaCy français en le mettant en cache."""

    global _SPACY_MODELE_NOM, _SPACY_MODELES_TENTES

    if spacy is None:
        _SPACY_MODELE_NOM = None
        _SPACY_MODELES_TENTES = ()
        return None

    essais: list[str] = []
    for nom_modele in ("fr_core_news_md", "fr_core_news_sm"):
        essais.append(nom_modele)
        try:
            modele = spacy.load(nom_modele)
        except OSError:
            continue
        except Exception:
            continue
        else:
            _SPACY_MODELE_NOM = nom_modele
            _SPACY_MODELES_TENTES = tuple(essais)
            return modele

    _SPACY_MODELE_NOM = None
    _SPACY_MODELES_TENTES = tuple(essais)
    return None


def _segmenter_en_phrases(texte: str) -> List[str]:
    """Segmente le texte en phrases en se basant sur la ponctuation forte."""
    if not texte:
        return []
    morceaux = re.split(r"(?<=[\.\!\?\:\;])\s+", texte)
    return [m.strip() for m in morceaux if m and m.strip()]


def _formater_noms_modeles(noms: Iterable[str]) -> str:
    noms_list = [f"'{nom}'" for nom in noms]
    if not noms_list:
        return ""
    if len(noms_list) == 1:
        return noms_list[0]
    return ", ".join(noms_list[:-1]) + f" et {noms_list[-1]}"


def _extraire_mots(
    phrase: str,
    *,
    longueur_min: int = 2,
    modele_spacy: Optional["Language"] = None,
) -> List[str]:
    """Extrait des mots en minuscule, filtrés sur la longueur minimale."""
    if not phrase:
        return []

    if modele_spacy is None:
        modele_spacy = _charger_modele_spacy()

    if modele_spacy is not None:
        doc = modele_spacy(phrase)
        tokens = []
        for token in doc:
            if not token.is_alpha:
                continue
            if len(token.text) < longueur_min:
                continue
            if token.is_stop:
                continue
            lemme = token.lemma_.lower() if token.lemma_ else token.text.lower()
            tokens.append(lemme)
        if tokens:
            return tokens

    mots = [m.lower() for m in _WORD_PATTERN.findall(phrase)]
    stopwords = SPACY_STOP_WORDS if SPACY_STOP_WORDS else set()
    return [m for m in mots if len(m) >= longueur_min and m not in stopwords]


def _mettre_en_evidence_mots(
    phrase: str,
    mots_cibles: Iterable[str],
    modele_spacy: Optional["Language"] = None,
) -> str:
    """Retourne la phrase en HTML avec les mots cibles entourés de <mark>."""

    mots_cibles_set = {m.lower() for m in mots_cibles if m}
    if not phrase or not mots_cibles_set:
        return html.escape(phrase)

    if modele_spacy is not None:
        try:
            doc = modele_spacy(phrase)
        except Exception:
            doc = None
    else:
        doc = None

    if doc is not None:
        morceaux: list[str] = []
        dernier_index = 0
        for token in doc:
            debut, fin = token.idx, token.idx + len(token.text)
            if debut >= len(phrase):
                continue
            morceaux.append(html.escape(phrase[dernier_index:debut]))
            lemme = token.lemma_.lower() if token.lemma_ else token.text.lower()
            texte_token = html.escape(token.text)
            if lemme in mots_cibles_set:
                morceaux.append(f"<mark>{texte_token}</mark>")
            else:
                morceaux.append(texte_token)
            dernier_index = fin
        morceaux.append(html.escape(phrase[dernier_index:]))
        return "".join(morceaux)

    morceaux: list[str] = []
    dernier_index = 0
    for match in _WORD_PATTERN.finditer(phrase):
        debut, fin = match.start(), match.end()
        morceaux.append(html.escape(phrase[dernier_index:debut]))
        mot = match.group(0)
        if mot.lower() in mots_cibles_set:
            morceaux.append(f"<mark>{html.escape(mot)}</mark>")
        else:
            morceaux.append(html.escape(mot))
        dernier_index = fin
    morceaux.append(html.escape(phrase[dernier_index:]))
    return "".join(morceaux)


def _generer_cooccurrences(
    texte: str,
    *,
    longueur_min: int = 2,
    modele_spacy: Optional["Language"] = None,
    granularite: str = "phrase",
) -> Counter[str]:
    """Compte les co-occurrences selon la granularité demandée."""

    compteurs: Counter[str] = Counter()

    if granularite == "document":
        tokens = _extraire_mots(
            texte,
            longueur_min=longueur_min,
            modele_spacy=modele_spacy,
        )
        if len(tokens) < 2:
            return compteurs
        frequences = Counter(tokens)
        mots_uniques = sorted(frequences.keys())
        for idx, mot1 in enumerate(mots_uniques):
            for mot2 in mots_uniques[idx + 1 :]:
                pair = f"{mot1}_{mot2}"
                compteurs[pair] = frequences[mot1] * frequences[mot2]
        return compteurs

    phrases = _segmenter_en_phrases(texte)
    for phrase in phrases:
        tokens = sorted(
            set(
                _extraire_mots(
                    phrase,
                    longueur_min=longueur_min,
                    modele_spacy=modele_spacy,
                )
            )
        )
        if len(tokens) < 2:
            continue
        for mot1, mot2 in combinations(tokens, 2):
            pair = f"{mot1}_{mot2}"
            compteurs[pair] += 1
    return compteurs


def calculer_table_cooccurrences(
    texte: str,
    *,
    longueur_min: int = 2,
    granularite: str = "phrase",
) -> pd.DataFrame:
    """Retourne un DataFrame des co-occurrences triées par fréquence décroissante."""
    modele_spacy = _charger_modele_spacy()
    compteur = _generer_cooccurrences(
        texte,
        longueur_min=longueur_min,
        modele_spacy=modele_spacy,
        granularite=granularite,
    )
    if not compteur:
        return pd.DataFrame(columns=["mot1", "mot2", "pair", "occurrences"])

    donnees = []
    for pair, freq in compteur.most_common():
        mot1, mot2 = pair.split("_", 1)
        donnees.append({
            "mot1": mot1,
            "mot2": mot2,
            "pair": pair,
            "occurrences": freq,
        })

    df = pd.DataFrame(donnees)
    df["occurrences"] = pd.to_numeric(df["occurrences"], errors="coerce").fillna(0).astype(int)
    return df


def _graphique_barres_cooccurrences(df: pd.DataFrame, top_n: int) -> alt.Chart | None:
    """Construit un graphique à barres Altair pour les co-occurrences les plus fréquentes."""
    if df.empty:
        return None

    top_df = df.head(top_n).copy()
    hauteur = max(300, 18 * len(top_df))
    chart = (
        alt.Chart(top_df)
        .mark_bar()
        .encode(
            x=alt.X("occurrences:Q", title="Occurrences"),
            y=alt.Y("pair:N", sort="-x", title="Paire de mots"),
            color=alt.Color("occurrences:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=[
                alt.Tooltip("mot1:N", title="Mot 1"),
                alt.Tooltip("mot2:N", title="Mot 2"),
                alt.Tooltip("occurrences:Q", title="Occurrences"),
            ],
        )
        .properties(height=hauteur)
    )
    return chart


def _nuage_de_mots(df: pd.DataFrame, max_mots: int) -> alt.Chart | None:
    """Construit un nuage de mots (mise en grille) pour visualiser les co-occurrences."""
    if df.empty:
        return None

    data = df.head(max_mots).copy()
    data["rang"] = range(len(data))
    if data.empty:
        return None

    nb_cols = max(4, int(math.sqrt(len(data))) or 1)
    data["colonne"] = data["rang"] % nb_cols
    data["ligne"] = -(data["rang"] // nb_cols)

    chart = (
        alt.Chart(data)
        .mark_text(baseline="middle", align="center")
        .encode(
            x=alt.X("colonne:Q", axis=None),
            y=alt.Y("ligne:Q", axis=None),
            text=alt.Text("pair:N"),
            size=alt.Size("occurrences:Q", scale=alt.Scale(range=[10, 80])),
            color=alt.Color("occurrences:Q", scale=alt.Scale(scheme="plasma"), legend=None),
            tooltip=[
                alt.Tooltip("pair:N", title="Co-occurrence"),
                alt.Tooltip("occurrences:Q", title="Occurrences"),
            ],
        )
        .configure_view(strokeWidth=0)
        .properties(width=700, height=400)
    )
    return chart


def _filtrer_cooccurrences_par_mot_cle(df: pd.DataFrame, mot_cle: str) -> pd.DataFrame:
    """Retourne les co-occurrences qui impliquent le mot-clé fourni."""

    mot_cle_normalise = mot_cle.strip().lower()
    if not mot_cle_normalise or df.empty:
        return pd.DataFrame(columns=[*df.columns, "mot_cle", "mot_associe"])

    masque_mot1 = df["mot1"].str.lower() == mot_cle_normalise
    masque_mot2 = df["mot2"].str.lower() == mot_cle_normalise
    masque = masque_mot1 | masque_mot2

    df_filtre = df.loc[masque].copy()
    if df_filtre.empty:
        return df_filtre

    df_filtre["mot_cle"] = mot_cle_normalise
    df_filtre["mot_associe"] = [
        ligne.mot2 if ligne.mot1.lower() == mot_cle_normalise else ligne.mot1
        for ligne in df_filtre.itertuples(index=False)
    ]
    return df_filtre


def render_cooccurrences_tab(texte_source: str) -> None:
    """Affiche l'onglet Streamlit consacré aux co-occurrences par mot-clé."""
    st.subheader("Analyse des co-occurrences par mot-clé")

    if not texte_source or not texte_source.strip():
        st.info("Saisissez ou chargez un texte pour analyser les co-occurrences.")
        return

    longueur_min = st.slider(
        "Longueur minimale des mots (en caractères)",
        min_value=1,
        max_value=6,
        value=2,
        help="Les mots plus courts que cette valeur sont ignorés pour stabiliser les co-occurrences.",
    )

    st.caption(
        "Les co-occurrences sont calculées phrase par phrase. Deux mots sont considérés "
        "comme co-occurrents s'ils apparaissent dans la même phrase."
    )

    modele_spacy = _charger_modele_spacy()
    if modele_spacy is None:
        if spacy is None:
            st.warning(
                "spaCy n'est pas disponible. Les stopwords ne seront pas filtrés."
            )
        else:
            noms_modeles = _SPACY_MODELES_TENTES or ("fr_core_news_md", "fr_core_news_sm")
            st.warning(
                "Les modèles spaCy "
                f"{_formater_noms_modeles(noms_modeles)} n'ont pas pu être chargés. "
                "Les stopwords ne seront pas filtrés."
            )
    elif _SPACY_MODELE_NOM:
        st.caption(f"Modèle spaCy chargé : {_SPACY_MODELE_NOM}")

    mot_cle_saisi = st.text_input(
        "Mot-clé pour une analyse ciblée",
        help="Analyse les co-occurrences limitées au mot indiqué.",
    )
    mot_cle_analyse = mot_cle_saisi.strip()
    if not mot_cle_analyse:
        st.info("Saisissez un mot-clé pour lancer l'analyse ciblée des co-occurrences.")
        return

    df_cooc = calculer_table_cooccurrences(
        texte_source,
        longueur_min=longueur_min,
        granularite="phrase",
    )

    if df_cooc.empty:
        st.info("Aucune co-occurrence n'a été détectée avec les paramètres actuels.")
        return

    df_mot_cle = _filtrer_cooccurrences_par_mot_cle(df_cooc, mot_cle_analyse)
    if df_mot_cle.empty:
        st.info(
            "Aucune co-occurrence ne contient le mot-clé « "
            f"{html.escape(mot_cle_analyse)} » avec les paramètres sélectionnés."
        )
        return

    max_occ = int(df_mot_cle["occurrences"].max())
    filtre_occ = st.slider(
        "Occurrences minimales",
        min_value=1,
        max_value=max_occ,
        value=1,
        help=(
            "Filtre les co-occurrences en fonction du nombre de phrases où elles apparaissent."
        ),
    )
    df_filtre = df_mot_cle[df_mot_cle["occurrences"] >= filtre_occ]

    if df_filtre.empty:
        st.info(
            "Le seuil minimal sélectionné exclut toutes les co-occurrences associées au mot-clé."
        )
        return

    st.markdown(
        "### Co-occurrences associées à « "
        f"{html.escape(mot_cle_analyse)} »"
    )
    st.dataframe(
        df_filtre[["mot_associe", "pair", "occurrences"]]
        .rename(columns={"mot_associe": "Mot associé"}),
        use_container_width=True,
        hide_index=True,
    )
    total_occurrences = int(df_filtre["occurrences"].sum())
    nb_associes = int(df_filtre["mot_associe"].nunique())
    st.caption(
        f"Le mot-clé apparaît dans {nb_associes} co-occurrence(s) distincte(s) "
        f"pour un total de {total_occurrences} occurrence(s)."
    )

    st.markdown("### Co-occurrences dans le texte")
    phrases = _segmenter_en_phrases(texte_source)
    if not phrases:
        st.info("Impossible de segmenter le texte en phrases pour afficher les co-occurrences.")
        return

    paires_filtrees = [
        (str(ligne.mot1), str(ligne.mot2))
        for ligne in df_filtre.itertuples(index=False)
    ]
    paires_uniques = list(dict.fromkeys(paires_filtrees))

    cooccurrences_trouvees = False
    for phrase in phrases:
        tokens_phrase = _extraire_mots(
            phrase,
            longueur_min=longueur_min,
            modele_spacy=modele_spacy,
        )
        if not tokens_phrase:
            continue
        tokens_set = set(tokens_phrase)
        paires_dans_phrase = [
            pair for pair in paires_uniques if pair[0] in tokens_set and pair[1] in tokens_set
        ]
        if not paires_dans_phrase:
            continue

        cooccurrences_trouvees = True
        mots_a_surligner = {mot for paire in paires_dans_phrase for mot in paire}
        phrase_html = _mettre_en_evidence_mots(
            phrase,
            mots_a_surligner,
            modele_spacy=modele_spacy,
        )
        st.markdown(
            f"<div style='margin-bottom:0.25rem'>{phrase_html}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Co-occurrences : "
            + ", ".join(f"{mot1} – {mot2}" for mot1, mot2 in paires_dans_phrase)
        )

    if not cooccurrences_trouvees:
        st.info(
            "Aucune des co-occurrences conservées ne figure explicitement dans les phrases."
        )
