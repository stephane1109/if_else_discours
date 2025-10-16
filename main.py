# -*- coding: utf-8 -*-
# Application Streamlit : discours politique → code (constructions Python existantes)
# + marqueurs normatifs + vues graphiques WHILE et IF en JPEG
# + annotation avec cases à cocher simples par famille IF/ELSE/WHILE/AND/OR
# + clés uniques pour tous les st.download_button
# + onglet « Co-occurrence » retiré, guide détaillé rétabli
# Auteur : Vous

import re
import json
import html
import difflib
import shutil
import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple, Any

# Graphviz (rendu JPEG)
try:
    import graphviz  # nécessite le binaire Graphviz installé (dot)
    GV_OK = shutil.which("dot") is not None
except Exception:
    GV_OK = False

# =========================
# Dictionnaires par défaut (familles Python réelles)
# =========================

DICTIONNAIRE_MOT_VERS_CODE_PAR_DEFAUT: Dict[str, str] = {
    # IF (conditions)
    "si": "IF",
    "s'il": "IF",
    "s'ils": "IF",
    "si l'on": "IF",
    "si on": "IF",
    "si jamais": "IF",
    "si seulement": "IF",
    "pourvu que": "IF",
    "à condition que": "IF",
    "a condition que": "IF",
    "à condition de": "IF",
    "a condition de": "IF",
    "à condition d'": "IF",
    "a condition d'": "IF",
    "sous réserve que": "IF",
    "sous reserve que": "IF",
    "sous condition que": "IF",
    "dans le cas où": "IF",
    "dans le cas ou": "IF",
    "au cas où": "IF",
    "au cas ou": "IF",
    "sous certaines conditions": "IF",

    # ELSE
    "sinon": "ELSE",

    # WHILE
    "tant que": "WHILE",

    # AND
    "et": "AND",
    "ainsi que": "AND",
    "de même que": "AND",
    "de meme que": "AND",
    "et aussi": "AND",

    # OR
    "ou": "OR",
    "ou bien": "OR",
    "ou alors": "OR",
    "soit": "OR",
}

MARQUEURS_PAR_DEFAUT: Dict[str, str] = {
    # OBLIGATION
    "il faut": "OBLIGATION",
    "il faut que": "OBLIGATION",
    "il faut de": "OBLIGATION",
    "nous devons": "OBLIGATION",
    "vous devez": "OBLIGATION",
    "doit": "OBLIGATION",
    "doivent": "OBLIGATION",
    "devra": "OBLIGATION",
    "devront": "OBLIGATION",
    "il faudra": "OBLIGATION",
    "il faudra que": "OBLIGATION",
    "il devra": "OBLIGATION",
    "il devra être": "OBLIGATION",
    "obligatoire": "OBLIGATION",
    "il s'impose": "OBLIGATION",
    "s'impose à": "OBLIGATION",
    "il appartient à": "OBLIGATION",
    "il revient à": "OBLIGATION",
    "il convient de": "OBLIGATION",
    "il y a lieu de": "OBLIGATION",
    "il y a urgence": "OBLIGATION",
    "il est urgent de": "OBLIGATION",
    "il est indispensable de": "OBLIGATION",
    "il est nécessaire de": "OBLIGATION",
    "il est temps de": "OBLIGATION",
    "à condition de": "OBLIGATION",
    "à condition que": "OBLIGATION",
    "sous certaines conditions": "OBLIGATION",

    # INTERDICTION
    "interdit": "INTERDICTION",
    "défense de": "INTERDICTION",
    "ne pas": "INTERDICTION",
    "il ne faut pas": "INTERDICTION",
    "il n'est pas possible de": "INTERDICTION",
    "ne peut pas": "INTERDICTION",
    "ne peuvent pas": "INTERDICTION",
    "ne pourra pas": "INTERDICTION",
    "ne pourront pas": "INTERDICTION",

    # PERMISSION
    "autorisé": "PERMISSION",
    "autorisés": "PERMISSION",
    "autorisées": "PERMISSION",
    "permis": "PERMISSION",
    "on peut": "PERMISSION",
    "il est possible de": "PERMISSION",
    "il peut": "PERMISSION",
    "peut": "PERMISSION",
    "peuvent": "PERMISSION",
    "pourra": "PERMISSION",
    "pourront": "PERMISSION",

    # RECOMMANDATION
    "il est recommandé": "RECOMMANDATION",
    "il serait souhaitable de": "RECOMMANDATION",
    "il est souhaitable de": "RECOMMANDATION",
    "devrait": "RECOMMANDATION",
    "devrions": "RECOMMANDATION",
    "conseillons": "RECOMMANDATION",
    "mieux vaut": "RECOMMANDATION",
    "il est bon de": "RECOMMANDATION",

    # SANCTION
    "sanction": "SANCTION",
    "sera sanctionné": "SANCTION",
    "seront sanctionnés": "SANCTION",
    "amende": "SANCTION",
    "peine": "SANCTION",
    "punition": "SANCTION",
    "coût de": "SANCTION",

    # CADRE
    "parlez librement": "CADRE_OUVERTURE",
    "ouverture du dialogue": "CADRE_OUVERTURE",
    "librement": "CADRE_OUVERTURE",
    "mieux vaut montrer une france unie": "CADRE_OUVERTURE",
    "pas de polémique": "CADRE_FERMETURE",
    "ce n’est pas le moment de débattre": "CADRE_FERMETURE",
    "ce n'est pas le moment de débattre": "CADRE_FERMETURE",
}

# =========================
# Représentation Python (étiquette affichée) et styles
# =========================

CODE_VERS_PYTHON: Dict[str, str] = {
    "IF": "if",
    "ELSE": "else",
    "WHILE": "while",
    "AND": "and",
    "OR": "or",
}

COULEURS_BADGES: Dict[str, Dict[str, str]] = {
    "IF": {"bg": "#e6f0ff", "fg": "#0b5ed7", "bd": "#0b5ed7"},
    "ELSE": {"bg": "#f3e8ff", "fg": "#6f42c1", "bd": "#6f42c1"},
    "WHILE": {"bg": "#e9fbe6", "fg": "#2f7d32", "bd": "#2f7d32"},
    "AND": {"bg": "#e6fffb", "fg": "#0d9488", "bd": "#0d9488"},
    "OR": {"bg": "#fff3e6", "fg": "#b54b00", "bd": "#b54b00"},
}

COULEURS_MARQUEURS: Dict[str, Dict[str, str]] = {
    "OBLIGATION": {"bg": "#fff7e6", "fg": "#a86600", "bd": "#a86600"},
    "INTERDICTION": {"bg": "#ffe6e6", "fg": "#c62828", "bd": "#c62828"},
    "PERMISSION": {"bg": "#e6fff3", "fg": "#2e7d32", "bd": "#2e7d32"},
    "RECOMMANDATION": {"bg": "#eef2ff", "fg": "#3f51b5", "bd": "#3f51b5"},
    "SANCTION": {"bg": "#fde7f3", "fg": "#ad1457", "bd": "#ad1457"},
    "CADRE_OUVERTURE": {"bg": "#e6f7ff", "fg": "#0277bd", "bd": "#0277bd"},
    "CADRE_FERMETURE": {"bg": "#ede7f6", "fg": "#6a1b9a", "bd": "#6a1b9a"},
}

# =========================
# Utilitaires texte / regex
# =========================

def normaliser_espace(texte: str) -> str:
    """Homogénéise espaces et apostrophes afin de stabiliser les recherches."""
    if not texte:
        return ""
    t = texte.replace("’", "'").replace("`", "'")
    t = re.sub(r"\s+", " ", t, flags=re.M)
    return t.strip()

def segmenter_en_phrases(texte: str) -> List[str]:
    """Segmente approximativement en phrases sur ponctuation forte."""
    if not texte:
        return []
    morceaux = re.split(r"(?<=[\.\!\?\:\;])\s+", texte)
    return [m.strip() for m in morceaux if m and m.strip()]

def construire_regex_depuis_liste(expressions: List[str]) -> List[Tuple[str, re.Pattern]]:
    """Construit des motifs regex en priorisant les locutions longues (gestion d'apostrophes)."""
    exprs_tries = sorted(expressions, key=lambda s: len(s), reverse=True)
    motifs = []
    for e in exprs_tries:
        e_norm = re.escape(e.replace("’", "'"))
        motif = re.compile(rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ]){e_norm}(?![A-Za-zÀ-ÖØ-öø-ÿ])", flags=re.I)
        motifs.append((e, motif))
    return motifs

# =========================
# I/O fichiers
# =========================

def lire_fichier_txt(uploaded_file) -> str:
    """Lit un fichier .txt avec stratégie automatique d’encodage (sans réglage UI)."""
    if uploaded_file is None:
        return ""
    donnees = uploaded_file.getvalue()
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return donnees.decode(enc)
        except Exception:
            continue
    return donnees.decode("utf-8", errors="ignore")

# =========================
# Détection (connecteurs / marqueurs)
# =========================

def detecter_connecteurs(texte: str, dico: Dict[str, str]) -> pd.DataFrame:
    """Renvoie un tableau des occurrences de connecteurs repérés dans le texte."""
    if not dico:
        return pd.DataFrame()
    texte_norm = normaliser_espace(texte)
    phrases = segmenter_en_phrases(texte_norm)
    motifs = construire_regex_depuis_liste(list(dico.keys()))
    enregs = []
    for i, ph in enumerate(phrases, start=1):
        ph_norm = normaliser_espace(ph)
        for c, motif in motifs:
            for m in motif.finditer(ph_norm):
                enregs.append({"id_phrase": i, "phrase": ph.strip(), "connecteur": c, "code": dico[c], "position": m.start()})
    df = pd.DataFrame(enregs)
    if not df.empty:
        df.sort_values(by=["id_phrase","position"], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)
    return df

def detecter_marqueurs_normatifs(texte: str, dico: Dict[str, str]) -> pd.DataFrame:
    """Renvoie un tableau des occurrences de marqueurs normatifs repérés dans le texte (avant ajustements)."""
    if not dico:
        return pd.DataFrame()
    texte_norm = normaliser_espace(texte)
    phrases = segmenter_en_phrases(texte_norm)
    motifs = construire_regex_depuis_liste(list(dico.keys()))
    enregs = []
    for i, ph in enumerate(phrases, start=1):
        ph_norm = normaliser_espace(ph)
        for cle, motif in motifs:
            for m in motif.finditer(ph_norm):
                enregs.append({
                    "id_phrase": i,
                    "phrase": ph.strip(),
                    "marqueur": cle,
                    "categorie": dico[cle],
                    "position": m.start(),
                    "longueur": m.end() - m.start()
                })
    df = pd.DataFrame(enregs)
    if not df.empty:
        df.sort_values(by=["id_phrase","position"], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)
    return df

# =========================
# Négation : ajustements regex uniquement
# =========================

def ajuster_negations_regex(phrase: str, dets_phrase: pd.DataFrame, treat_ne_sans_pas_as_neg: bool = False) -> pd.DataFrame:
    """
    Reclasse en INTERDICTION les formes 'ne … peut/peuvent/pourra/pourront (pas/plus/jamais)'.
    Optionnel : traiter 'ne … peut' même SANS 'pas' comme négation (ellipse), sauf 'peut-être'.
    """
    if dets_phrase.empty:
        return dets_phrase

    patrons = [
        r"\bne\s+\w{0,2}\s*peut\b(?:\s+\w+){0,6}?(?:pas|plus|jamais)\b",
        r"\bne\s+\w{0,2}\s*peuvent\b(?:\s+\w+){0,6}?(?:pas|plus|jamais)\b",
        r"\bne\s+\w{0,2}\s*pourra\b(?:\s+\w+){0,6}?(?:pas|plus|jamais)\b",
        r"\bne\s+\w{0,2}\s*pourront\b(?:\s+\w+){0,6}?(?:pas|plus|jamais)\b",
    ]
    spans_neg = []
    for pat in patrons:
        for m in re.finditer(pat, phrase, flags=re.I):
            spans_neg.append((m.start(), m.end()))

    if treat_ne_sans_pas_as_neg:
        for m in re.finditer(r"\bne\s+\w{0,2}\s*peut\b(?![-\s]*être)\b", phrase, flags=re.I):
            fin = len(phrase)
            virg = re.search(r"[,;:\.\!\?]", phrase[m.end():])
            if virg:
                fin = m.end() + virg.start()
            spans_neg.append((m.start(), fin))

    if not spans_neg:
        return dets_phrase

    def chevauche(pos, longu, spans):
        debut, fin = pos, pos + longu
        for s, e in spans:
            if not (fin < s or debut > e):
                return True
        return False

    dets = dets_phrase.copy()
    cible = dets["marqueur"].str.lower().isin(["peut", "peuvent", "pourra", "pourront", "il peut"])
    for idx, row in dets.loc[cible].iterrows():
        if chevauche(row["position"], row["longueur"], spans_neg):
            dets.at[idx, "categorie"] = "INTERDICTION"
            if row["marqueur"].lower() == "il peut":
                dets.at[idx, "marqueur"] = "il ne peut (négation)"
            else:
                dets.at[idx, "marqueur"] = f"ne {row['marqueur']} (négation)"
    return dets

def ajuster_negations_global(texte: str, df_norm: pd.DataFrame, treat_ne_sans_pas_as_neg: bool = False) -> pd.DataFrame:
    """Applique les ajustements de négation phrase par phrase uniquement via regex."""
    if df_norm.empty:
        return df_norm
    phrases = segmenter_en_phrases(texte)
    dets_list = []
    for i, ph in enumerate(phrases, start=1):
        bloc = df_norm[df_norm["id_phrase"] == i].copy()
        if bloc.empty:
            dets_list.append(bloc); continue
        bloc = ajuster_negations_regex(ph, bloc, treat_ne_sans_pas_as_neg=treat_ne_sans_pas_as_neg)
        dets_list.append(bloc)
    df_adj = pd.concat(dets_list, ignore_index=True) if dets_list else df_norm
    if not df_adj.empty:
        df_adj.sort_values(by=["id_phrase","position"], inplace=True, kind="mergesort")
        df_adj.reset_index(drop=True, inplace=True)
    return df_adj

# =========================
# Annotation HTML (texte + badges) — filtres simples par famille
# =========================

def _esc(s: str) -> str:
    """Échappe le HTML pour un affichage sûr."""
    return html.escape(s, quote=False)

def css_badges() -> str:
    """CSS pour les badges et styles d’annotation."""
    lignes = [
        "<style>",
        ".texte-annote { line-height: 1.8; font-size: 1.05rem; white-space: pre-wrap; }",
        ".badge-code { display: inline-block; padding: 0.05rem 0.4rem; margin-left: 0.25rem; border: 1px solid #333; border-radius: 0.35rem; font-family: monospace; font-size: 0.85em; vertical-align: baseline; }",
        ".badge-marqueur { display: inline-block; padding: 0.03rem 0.35rem; margin-left: 0.2rem; border: 1px dashed #333; border-radius: 0.35rem; font-family: monospace; font-size: 0.78em; vertical-align: baseline; }",
        ".connecteur { font-weight: 600; }",
        ".mot-marque { font-weight: 600; text-decoration: underline; }",
        "</style>",
    ]
    for code, pal in COULEURS_BADGES.items():
        lignes.insert(-1, f".badge-code.code-{code} {{ background-color: {pal['bg']}; color: {pal['fg']}; border-color: {pal['bd']}; }}")
    for cat, pal in COULEURS_MARQUEURS.items():
        lignes.insert(-1, f".badge-marqueur.marq-{_esc(cat)} {{ background-color: {pal['bg']}; color: {pal['fg']}; border-color: {pal['bd']}; }}")
    return "\n".join(lignes)

def occurrences_mixte(texte: str,
                      dico_conn: Dict[str, str],
                      dico_norm: Dict[str, str]) -> List[Dict[str, Any]]:
    """Retourne toutes les occurrences (connecteurs + marqueurs), avec élimination des chevauchements."""
    occs: List[Dict[str, Any]] = []
    for type_lib, dico in [("connecteur", dico_conn), ("marqueur", dico_norm)]:
        motifs = construire_regex_depuis_liste(list(dico.keys()))
        for cle, motif in motifs:
            for m in motif.finditer(texte):
                etiquette = dico[cle]
                occs.append({
                    "debut": m.start(), "fin": m.end(), "original": texte[m.start():m.end()],
                    "type": type_lib, "cle": cle, "etiquette": etiquette, "longueur": m.end()-m.start(),
                })
    occs.sort(key=lambda x: (x["debut"], -x["longueur"]))
    res = []; borne = -1
    for m in occs:
        if m["debut"] >= borne:
            res.append(m); borne = m["fin"]
    return res

def libelle_python(code: str) -> str:
    """Retourne l’étiquette Python à afficher."""
    return CODE_VERS_PYTHON.get(str(code).upper(), str(code).upper())

def html_annote(texte: str,
                dico_conn: Dict[str, str],
                dico_norm: Dict[str, str],
                show_codes: Dict[str, bool],
                show_marqueurs: bool,
                filtres_marqueurs: List[str] = None) -> str:
    """
    Produit le HTML annoté où l’étiquette Python colorée est placée à côté du mot français.
    show_codes : dict {'IF':True/False, 'ELSE':..., 'WHILE':..., 'AND':..., 'OR':...}
    show_marqueurs : afficher ou non les marqueurs normatifs
    filtres_marqueurs : listes de catégories à garder (si vide → toutes)
    """
    if not texte:
        return "<div class='texte-annote'>(Texte vide)</div>"
    filtres_marqueurs = set([c.upper() for c in (filtres_marqueurs or [])])

    t = texte
    occ = occurrences_mixte(t, dico_conn, dico_norm)
    morceaux: List[str] = []; curseur = 0
    for m in occ:
        if m["type"] == "connecteur":
            code = str(m["etiquette"]).upper()
            if not show_codes.get(code, True):
                continue
        else:
            if not show_marqueurs:
                continue
            cat = str(m["etiquette"]).upper()
            if filtres_marqueurs and cat not in filtres_marqueurs:
                continue

        if m["debut"] > curseur:
            morceaux.append(_esc(t[curseur:m["debut"]]))
        mot_original = _esc(t[m["debut"]:m["fin"]])
        if m["type"] == "connecteur":
            code = str(m["etiquette"]).upper()
            badge = f"<span class='badge-code code-{_esc(code)}'>{_esc(libelle_python(code))}</span>"
            rendu = f"<span class='connecteur'>{mot_original}</span>{badge}"
        else:
            cat = str(m["etiquette"]).upper()
            badge = f"<span class='badge-marqueur marq-{_esc(cat)}'>{_esc(cat)}</span>"
            rendu = f"<span class='mot-marque'>{mot_original}</span>{badge}"
        morceaux.append(rendu); curseur = m["fin"]
    if curseur < len(t):
        morceaux.append(_esc(t[curseur:]))

    if not morceaux:
        return "<div class='texte-annote'>(Aucune annotation selon les cases sélectionnées)</div>"
    return "<div class='texte-annote'>" + "".join(morceaux) + "</div>"

def html_autonome(fragment_html: str) -> str:
    """Génère un document HTML autonome avec CSS embarqué."""
    return f"<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'/><title>Texte annoté</title>{css_badges()}</head><body>{fragment_html}</body></html>"

# =========================
# Couverture du lexique et candidats
# =========================

def classer_candidat_code(expr: str) -> str:
    """Heuristique pour deviner la famille Python (IF/ELSE/WHILE/AND/OR) d’un candidat."""
    e = expr.lower()
    if re.search(r"\b(et|ainsi que|de même que|de meme que|et aussi)\b", e): return "AND"
    if re.search(r"\b(ou|ou bien|ou alors|soit)\b", e): return "OR"
    if re.search(r"\btant que\b", e): return "WHILE"
    if re.search(r"\bsinon\b", e): return "ELSE"
    if re.search(r"\b(si|s['’]il|s['’]ils|si l'on|si on|si jamais|si seulement|pourvu que|à condition|a condition|sous r[ée]serve|sous condition|au cas o[uù]|dans le cas o[uù])\b", e): return "IF"
    return ""

def extraire_candidats_connecteurs(texte: str) -> List[str]:
    """Extrait des locutions candidates liées aux familles Python gérées."""
    t = " " + normaliser_espace(texte) + " "
    candidats = set()
    patrons = [
        # IF
        r"\bs['’]?il[s]?\b(?:\s+\w+){0,2}",
        r"\bsi\b(?:\s+\w+){0,6}",
        r"\bsi\s+jamais\b(?:\s+\w+){0,3}",
        r"\bsi\s+seulement\b(?:\s+\w+){0,3}",
        r"\bpourvu\s+que\b(?:\s+\w+){0,3}",
        r"\bà\s+condition\s+que\b(?:\s+\w+){0,3}",
        r"\bà\s+condition\s+d['’]?\b(?:\w+\s*){0,3}",
        r"\bsous\s+r[ée]serve\s+que\b(?:\s+\w+){0,3}",
        r"\bsous\s+condition\s+que\b(?:\s+\w+){0,3}",
        r"\bau\s+cas\s+o[uù]\b(?:\s+\w+){0,3}",
        r"\bdans\s+le\s+cas\s+o[uù]\b(?:\s+\w+){0,3}",
        # ELSE
        r"\bsinon\b",
        # WHILE
        r"\btant\s+que\b(?:\s+\w+){0,6}",
        # AND
        r"\bet\b",
        r"\bainsi\s+que\b",
        r"\bde\s+m[eè]me\s+que\b",
        r"\bet\s+aussi\b",
        # OR
        r"\bou\b",
        r"\bou\s+bien\b",
        r"\bou\s+alors\b",
        r"\bsoit\b",
    ]
    for pat in patrons:
        for m in re.finditer(pat, t, flags=re.I):
            expr = m.group(0).strip()
            toks = expr.split()
            if len(toks) > 7:
                expr = " ".join(toks[:7])
            candidats.add(expr.lower())
    return sorted(candidats)

def rapport_couverture_lexique(dico_conn: Dict[str, str], df_conn: pd.DataFrame, texte: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (1) couverture par famille ; (2) candidats proches non reconnus."""
    code_to_keys = {}
    for k, v in dico_conn.items():
        code_to_keys.setdefault(v, set()).add(k)
    observes_uniques = set(df_conn["connecteur"].str.lower()) if not df_conn.empty else set()
    familles = ["IF","ELSE","WHILE","AND","OR"]

    lignes = []
    for code in familles:
        keys = code_to_keys.get(code, set())
        connus = len(keys)
        observes = len(keys & observes_uniques)
        non_obs = max(0, connus - observes)
        lignes.append({"code": code, "connus": connus, "observes": observes, "non_observes": non_obs})
    df_couv = pd.DataFrame(lignes)

    candidats = set(extraire_candidats_connecteurs(texte))
    candidats = [c for c in candidats if c not in dico_conn]
    enregs = []
    for cand in candidats:
        code_guess = classer_candidat_code(cand)
        if not code_guess:
            continue
        ref_keys = sorted(list(code_to_keys.get(code_guess, set())))
        best = ""
        score_best = 0.0
        for ref in ref_keys:
            s = difflib.SequenceMatcher(None, cand, ref).ratio()
            if s > score_best:
                score_best = s; best = ref
        enregs.append({
            "candidat": cand,
            "code_suggere": code_guess,
            "cle_proche": best,
            "similarite": round(score_best, 3)
        })
    df_candidats = pd.DataFrame(enregs).sort_values(["code_suggere", "similarite"], ascending=[True, False]).reset_index(drop=True)
    return df_candidats, df_couv  # (on gardera l’ordre d’usage dans l’onglet)

# =========================
# Vues graphiques : WHILE et IF (afficher TOUTES les occurrences) + export JPEG
# =========================

def extraire_segments_while(texte: str) -> List[Dict[str, Any]]:
    """Extrait les occurrences de 'tant que …' et sépare grossièrement condition/action sur la 1re ponctuation faible."""
    res = []
    phrases = segmenter_en_phrases(texte)
    for idx, ph in enumerate(phrases, start=1):
        for m in re.finditer(r"\btant\s+que\b", ph, flags=re.I):
            suite = ph[m.end():].strip()
            cut = re.split(r"[,;:\-\—]\s+", suite, maxsplit=1)
            condition = cut[0].strip() if cut and cut[0] else suite
            action = cut[1].strip() if cut and len(cut) > 1 else ""
            res.append({
                "type": "WHILE",
                "id_phrase": idx,
                "phrase": ph.strip(),
                "condition": condition,
                "action_true": action,
                "action_false": "",
                "debut": m.start(),
            })
    return res

def extraire_segments_if(texte: str) -> List[Dict[str, Any]]:
    """Extrait les occurrences de 'si … [alors …] [sinon …]' avec découpes simples."""
    res = []
    phrases = segmenter_en_phrases(texte)
    for idx, ph in enumerate(phrases, start=1):
        for m in re.finditer(r"\bsi\b(?:\s+l['’]on|\s+on|\s+jamais|\s+seulement)?|\bs['’]il(?:s)?\b", ph, flags=re.I):
            suite = ph[m.end():].strip()
            m_alors = re.search(r"\balors\b", suite, flags=re.I)
            if m_alors:
                condition = suite[:m_alors.start()].strip(" ,;:-—")
                reste = suite[m_alors.end():].strip()
            else:
                cut = re.split(r"[,;:\-\—]\s+", suite, maxsplit=1)
                condition = cut[0].strip() if cut else suite
                reste = cut[1].strip() if len(cut) > 1 else ""

            action_true = reste
            action_false = ""
            m_sinon = re.search(r"\bsinon\b", reste, flags=re.I)
            if m_sinon:
                action_true = reste[:m_sinon.start()].strip(" ,;:-—")
                action_false = reste[m_sinon.end():].strip(" .;:-—")

            res.append({
                "type": "IF",
                "id_phrase": idx,
                "phrase": ph.strip(),
                "condition": condition.strip(" ."),
                "action_true": action_true.strip(" ."),
                "action_false": action_false.strip(" ."),
                "debut": m.start(),
            })
    return res

def graphviz_while_dot(condition: str, action: str) -> str:
    """Construit un code DOT simple pour WHILE."""
    def esc(s: str) -> str:
        return s.replace('"', r"\"")
    cond_txt = esc(condition if condition else "(condition non extraite)")
    act_txt = esc(action if action else "(action implicite ou non extraite)")
    dot = f'''
digraph G {{
  rankdir=LR;
  node [shape=box, fontname="Helvetica"];
  start [shape=circle, label="Start"];
  cond [shape=diamond, label="while ({cond_txt})"];
  act  [shape=box, label="{act_txt}"];
  end  [shape=doublecircle, label="End"];

  start -> cond;
  cond -> act [label="True"];
  act  -> cond [label="itère"];
  cond -> end [label="False"];
}}
'''
    return dot

def graphviz_if_dot(condition: str, action_true: str, action_false: str = "") -> str:
    """Construit un code DOT pour IF/ELSE."""
    def esc(s: str) -> str:
        return s.replace('"', r"\"")
    cond_txt = esc(condition if condition else "(condition non extraite)")
    act_t = esc(action_true if action_true else "(action si vrai non extraite)")
    act_f = esc(action_false) if action_false else ""
    has_else = bool(action_false.strip())

    dot = [
        "digraph G {",
        "  rankdir=LR;",
        '  node [shape=box, fontname="Helvetica"];',
        '  start [shape=circle, label="Start"];',
        f'  cond  [shape=diamond, label="if ({cond_txt})"];',
        f'  actt  [shape=box, label="{act_t}"];',
    ]
    if has_else:
        dot.append(f'  actf  [shape=box, label="{act_f}"];')
    dot.append('  end   [shape=doublecircle, label="End"];')
    dot += [
        "  start -> cond;",
        '  cond  -> actt [label="True"];',
        "  actt  -> end;",
    ]
    if has_else:
        dot += [
            '  cond  -> actf [label="False"];',
            "  actf  -> end;",
        ]
    dot.append("}")
    return "\n".join(dot)

def rendre_jpeg_depuis_dot(dot_str: str) -> bytes:
    """Rend un graphe DOT en JPEG (octets). Nécessite Graphviz installé."""
    if not GV_OK:
        raise RuntimeError("Graphviz (binaire 'dot') indisponible sur ce système.")
    src = graphviz.Source(dot_str)
    return src.pipe(format="jpg")

# =========================
# État / Session (initialisation)
# =========================

if "dico_mot_vers_code" not in st.session_state:
    st.session_state["dico_mot_vers_code"] = DICTIONNAIRE_MOT_VERS_CODE_PAR_DEFAUT.copy()
if "dico_marqueurs" not in st.session_state:
    st.session_state["dico_marqueurs"] = MARQUEURS_PAR_DEFAUT.copy()

# =========================
# Interface
# =========================

st.set_page_config(page_title="Discours → Code (Python) & Marqueurs normatifs", page_icon=None, layout="wide")
st.title("Analyse d’un discours comme du code : IF / ELSE / WHILE / AND / OR + marqueurs (regex)")

with st.sidebar:
    st.header("Options d’analyse")
    opt_ne_sans_pas = st.checkbox("Considérer « ne … peut » sans « pas » comme négation", value=False,
                                  help="Utile si le texte a perdu des « pas » ; ignore « peut-être ».")
    st.markdown("---")

    st.header("Connecteurs (expression française → famille Python)")
    dico_conn = st.session_state["dico_mot_vers_code"]
    nouv_mot = st.text_input("Ajouter une expression (connecteur)")
    nouv_code = st.selectbox("Famille Python", ["IF","ELSE","WHILE","AND","OR"])
    if st.button("Ajouter le connecteur"):
        if nouv_mot:
            cle = normaliser_espace(nouv_mot.lower())
            dico_conn[cle] = str(nouv_code).upper()
            st.session_state["dico_mot_vers_code"] = dico_conn
            st.success(f"Ajouté : « {nouv_mot} » → {nouv_code}")

    st.download_button("Télécharger connecteurs (JSON)",
                       data=json.dumps(st.session_state["dico_mot_vers_code"], ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="dictionnaire_connecteurs.json", mime="application/json",
                       key="dl_conn_json_sidebar")

    st.divider()
    st.header("Marqueurs normatifs (dictionnaire enrichi)")
    dico_norm = st.session_state["dico_marqueurs"]
    nouv_marq = st.text_input("Ajouter une expression (normatif)")
    cat_marq = st.selectbox("Catégorie", ["OBLIGATION","INTERDICTION","PERMISSION","RECOMMANDATION","SANCTION","CADRE_OUVERTURE","CADRE_FERMETURE"])
    if st.button("Ajouter le marqueur normatif"):
        if nouv_marq:
            cle = normaliser_espace(nouv_marq.lower())
            dico_norm[cle] = str(cat_marq).upper()
            st.session_state["dico_marqueurs"] = dico_norm
            st.success(f"Ajouté : « {nouv_marq} » → {cat_marq}")

    st.download_button("Télécharger marqueurs (JSON)",
                       data=json.dumps(st.session_state["dico_marqueurs"], ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="dictionnaire_marqueurs.json", mime="application/json",
                       key="dl_marq_json_sidebar")

# Source du discours
st.markdown("### Source du discours")
mode_source = st.radio("Choisir la source du texte", options=["Fichier .txt", "Zone de texte"], index=0, horizontal=True)

texte_source = ""
if mode_source == "Fichier .txt":
    fichier_txt = st.file_uploader("Déposer un fichier texte (.txt)", type=["txt"], accept_multiple_files=False, key="discours_txt")
    if fichier_txt is not None:
        try:
            texte_source = lire_fichier_txt(fichier_txt)
            st.success(f"Fichier chargé : {fichier_txt.name} • {len(texte_source)} caractères")
        except Exception as e:
            st.error(f"Impossible de lire le fichier : {e}")
else:
    texte_source = st.text_area("Saisir ou coller le discours :", value="", height=240, placeholder="Coller ici le discours à analyser…")

st.divider()

# Préparer les détections
if texte_source.strip():
    df_conn = detecter_connecteurs(texte_source, st.session_state["dico_mot_vers_code"])
    df_norm_brut = detecter_marqueurs_normatifs(texte_source, st.session_state["dico_marqueurs"])
    df_norm = ajuster_negations_global(texte_source, df_norm_brut, treat_ne_sans_pas_as_neg=opt_ne_sans_pas)
else:
    df_conn = pd.DataFrame()
    df_norm = pd.DataFrame()

# Onglets (co-occurrence retiré)
ong1, ong2, ong3, ong4, ong5, ong6 = st.tabs([
    "Expressions mappées", "Détections", "Couverture du lexique", "Dictionnaires (JSON)", "Guide d’interprétation", "Graphiques (IF / WHILE)"
])

# Onglet 1 : Expressions mappées
with ong1:
    st.subheader("Expressions françaises mappées vers une famille Python (if / else / while / and / or)")
    dico_conn = st.session_state["dico_mot_vers_code"]
    if not dico_conn:
        st.info("Le dictionnaire des connecteurs est vide.")
    else:
        df_map = pd.DataFrame(sorted([(k, v) for k, v in dico_conn.items()], key=lambda x: (x[1], x[0])),
                              columns=["expression_française", "famille_python"])
        st.dataframe(df_map, use_container_width=True, hide_index=True)
        st.download_button("Exporter le mappage (CSV)",
                           data=df_map.to_csv(index=False).encode("utf-8"),
                           file_name="mapping_connecteurs.csv", mime="text/csv",
                           key="dl_map_csv")

    st.subheader("Explication des marqueurs normatifs (dictionnaire enrichi)")
    st.write("Les marqueurs normatifs détectent les segments prescriptifs : OBLIGATION, INTERDICTION, PERMISSION, RECOMMANDATION, SANCTION, ainsi que l’ouverture/fermeture du cadre de débat.")

# Onglet 2 : Détections + texte annoté (cases simples par famille)
with ong2:
    st.subheader("Connecteurs détectés")
    if df_conn.empty:
        st.info("Aucun connecteur détecté ou aucun texte fourni.")
    else:
        st.dataframe(df_conn, use_container_width=True, hide_index=True)
        st.download_button("Exporter connecteurs (CSV)",
                           data=df_conn.to_csv(index=False).encode("utf-8"),
                           file_name="occurrences_connecteurs.csv", mime="text/csv",
                           key="dl_occ_conn_csv")

    st.subheader("Marqueurs normatifs détectés (après ajustements)")
    if df_norm.empty:
        st.info("Aucun marqueur normatif détecté ou aucun texte fourni.")
    else:
        st.dataframe(df_norm, use_container_width=True, hide_index=True)
        st.download_button("Exporter marqueurs normatifs (CSV)",
                           data=df_norm.to_csv(index=False).encode("utf-8"),
                           file_name="occurrences_marqueurs_normatifs.csv", mime="text/csv",
                           key="dl_occ_marq_csv")

    st.markdown("---")
    st.subheader("Texte annoté : cases à cocher simples par **famille**")

    colA, colB, colC, colD, colE = st.columns(5)
    with colA:
        show_if = st.checkbox("IF (if)", value=True)
    with colB:
        show_else = st.checkbox("ELSE (else)", value=True)
    with colC:
        show_while = st.checkbox("WHILE (while)", value=True)
    with colD:
        show_and = st.checkbox("AND (and)", value=True)
    with colE:
        show_or = st.checkbox("OR (or)", value=True)

    show_marqueurs = st.checkbox("Afficher les marqueurs normatifs", value=True)
    filt_marq = st.multiselect("Limiter aux catégories de marqueurs (optionnel)",
                               ["OBLIGATION","INTERDICTION","PERMISSION","RECOMMANDATION","SANCTION","CADRE_OUVERTURE","CADRE_FERMETURE"],
                               default=[])

    show_codes = {
        "IF": show_if,
        "ELSE": show_else,
        "WHILE": show_while,
        "AND": show_and,
        "OR": show_or,
    }

    st.markdown(css_badges(), unsafe_allow_html=True)
    if not texte_source.strip():
        st.info("Aucun texte fourni.")
        frag = "<div class='texte-annote'>(Texte vide)</div>"
    else:
        frag = html_annote(
            texte_source,
            st.session_state["dico_mot_vers_code"],
            st.session_state["dico_marqueurs"],
            show_codes=show_codes,
            show_marqueurs=show_marqueurs,
            filtres_marqueurs=filt_marq
        )
        st.markdown(frag, unsafe_allow_html=True)
        st.download_button("Exporter le texte annoté (HTML)",
                           data=html_autonome(frag).encode("utf-8"),
                           file_name="texte_annote.html", mime="text/html",
                           key="dl_annote_html")

# Onglet 3 : Couverture du lexique
with ong3:
    st.subheader("Couverture du lexique (connecteurs Python) et candidats proches non reconnus")
    if not texte_source.strip():
        st.info("Aucun texte fourni.")
    else:
        df_candidats, df_couv = rapport_couverture_lexique(st.session_state["dico_mot_vers_code"], df_conn, texte_source)
        st.markdown("Résumé par famille de connecteurs")
        st.dataframe(df_couv, use_container_width=True, hide_index=True)
        st.download_button("Exporter couverture (CSV)",
                           data=df_couv.to_csv(index=False).encode("utf-8"),
                           file_name="couverture_lexique_connecteurs.csv", mime="text/csv",
                           key="dl_couv_csv")
        st.markdown("Candidats proches non reconnus (propositions d’enrichissement du dictionnaire)")
        if df_candidats.empty:
            st.info("Aucun candidat proche détecté avec les heuristiques actuelles.")
        else:
            st.dataframe(df_candidats, use_container_width=True, hide_index=True)
            st.download_button("Exporter candidats (CSV)",
                               data=df_candidats.to_csv(index=False).encode("utf-8"),
                               file_name="candidats_connecteurs.csv", mime="text/csv",
                               key="dl_cand_csv")

# Onglet 4 : Dictionnaires (JSON)
with ong4:
    sous1, sous2 = st.tabs(["Connecteurs (JSON)", "Marqueurs normatifs (JSON)"])
    with sous1:
        st.json(st.session_state["dico_mot_vers_code"], expanded=False)
        st.download_button("Télécharger connecteurs (JSON)",
                           data=json.dumps(st.session_state["dico_mot_vers_code"], ensure_ascii=False, indent=2).encode("utf-8"),
                           file_name="dictionnaire_connecteurs.json", mime="application/json",
                           key="dl_conn_json_tab")
    with sous2:
        st.json(st.session_state["dico_marqueurs"], expanded=False)
        st.download_button("Télécharger marqueurs (JSON)",
                           data=json.dumps(st.session_state["dico_marqueurs"], ensure_ascii=False, indent=2).encode("utf-8"),
                           file_name="dictionnaire_marqueurs.json", mime="application/json",
                           key="dl_marq_json_tab")

# Onglet 5 : Guide d’interprétation (détaillé)
with ong5:
    st.subheader("Guide d’interprétation : Python vs analyse de discours")

    st.markdown("#### IF")
    st.write(
        "Cadre Python : `if` introduit une condition booléenne ; le bloc indenté s’exécute si la condition est vraie. "
        "On enchaîne souvent `elif` (autres cas) puis `else` (cas par défaut). "
        "En Python, la condition doit évaluer à `True`/`False` (valeurs dites truthy/falsy admises). "
        "Exemples : `if x > 0:`, `if user and is_admin:`.\n\n"
        "Cadre analyse : « si », « à condition que », « pourvu que », « au cas où », etc. posent une **condition d’acceptabilité** "
        "ou de **mise en œuvre**. Elles délimitent ce qui doit être vrai pour autoriser la suite de l’énoncé (promesse, action, engagement). "
        "Repérer ces formes permet d’identifier les clauses de contingence (quand/à quelles conditions on agit)."
    )

    st.markdown("#### ELSE")
    st.write(
        "Cadre Python : `else` est la branche alternative quand la condition précédente est fausse. "
        "Elle n’a pas de condition propre ; elle couvre le « tout le reste ».\n\n"
        "Cadre analyse : « sinon » introduit l’**alternative par défaut** ou un **coût narratif** si la condition échoue. "
        "Cela structure le discours en **deux issues** distinctes (acceptation vs rejet de la condition)."
    )

    st.markdown("#### WHILE")
    st.write(
        "Cadre Python : `while` répète un bloc tant que l’expression reste vraie ; le corps peut comporter `break` (sortie) ou `continue` (saut à l’itération suivante). "
        "Attention aux boucles infinies si la condition ne change pas.\n\n"
        "Cadre analyse : « tant que » exprime une **persistance conditionnelle** (maintien d’une action ou d’une politique tant qu’un état perdure). "
        "On modélise ainsi la **durée** des engagements : quelles actions se maintiennent et à quelles conditions cessent-elles ?"
    )

    st.markdown("#### AND")
    st.write(
        "Cadre Python : `and` évalue en court-circuit (si la première sous-condition est fausse, la seconde n’est pas évaluée). "
        "Il exige que **toutes** les sous-conditions soient vraies pour que l’ensemble soit vrai. "
        "Exemples : `if budget_ok and votes >= seuil:`.\n\n"
        "Cadre analyse : « et », « ainsi que », « de même que » **agrègent** des éléments, des causes ou des engagements pour construire un **front commun**. "
        "Ils servent à cumuler des critères (tous requis), à densifier l’argumentaire et à afficher des coalitions (X **et** Y)."
    )

    st.markdown("#### OR")
    st.write(
        "Cadre Python : `or` est vrai si **au moins** une sous-condition est vraie (court-circuit si la première est vraie). "
        "Exemples : `if crise or urgence:`.\n\n"
        "Cadre analyse : « ou », « ou bien », « soit » posent des **alternatives**. "
        "L’usage répétitif de `ou` peut signaler une **délégation du choix** à l’auditoire ou une **flexibilité** stratégique."
    )

    st.markdown("#### Négation et verbe « pouvoir »")
    st.write(
        "Le script reclassifie automatiquement les constructions **« ne … peut/peuvent/pourra/pourront (pas/plus/jamais) »** en **INTERDICTION**. "
        "Option disponible pour traiter aussi l’ellipse « ne … peut » sans « pas » (sauf « peut-être ») lorsque des pertes de mots négatifs sont suspectées. "
        "Dans l’analyse, ces formes modulent la **permission** : de la permission explicite (« on peut ») à la **fermeture** (« on ne peut pas »)."
    )

    st.markdown("#### Marqueurs normatifs")
    st.write(
        "Les catégories OBLIGATION / INTERDICTION / PERMISSION / RECOMMANDATION / SANCTION / CADRE_OUVERTURE / CADRE_FERMETURE guident la lecture pragmatique : "
        "**qui contraint qui**, **qu’est-ce qui est interdit/autorisé**, **quels coûts/menaces** sont attachés à l’inaction, "
        "et **quel cadre d’échange** est promu (ou fermé). Leur co-présence avec des `if/while` éclaire les stratégies d’engagement conditionnel."
    )

# Onglet 6 : Graphiques (IF / WHILE) — tout afficher, export JPEG
with ong6:
    st.subheader("Boucles WHILE détectées")
    if not texte_source.strip():
        st.info("Aucun texte fourni.")
    else:
        seg_while = extraire_segments_while(texte_source)
        if not seg_while:
            st.info("Aucune occurrence « tant que … » détectée.")
        else:
            for i, sel in enumerate(seg_while, start=1):
                st.markdown(f"**WHILE — phrase {sel['id_phrase']}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("Condition")
                    st.write(sel["condition"] if sel["condition"] else "(non extraite)")
                with col2:
                    st.markdown("Action (itération)")
                    st.write(sel["action_true"] if sel["action_true"] else "(implicite ou non extraite)")

                dot = graphviz_while_dot(sel["condition"], sel["action_true"])

                if GV_OK:
                    try:
                        img_bytes = rendre_jpeg_depuis_dot(dot)
                        st.image(img_bytes, caption=f"Graphe WHILE — phrase {sel['id_phrase']}", use_container_width=True)
                        st.download_button(f"Télécharger le graphe WHILE #{sel['id_phrase']} (JPEG)",
                                           data=img_bytes,
                                           file_name=f"while_phrase_{sel['id_phrase']}.jpg",
                                           mime="image/jpeg",
                                           key=f"dl_while_jpg_{sel['id_phrase']}_{i}")
                    except Exception as e:
                        st.error(f"Rendu JPEG indisponible : {e}")
                        st.code(dot, language="dot")
                else:
                    st.error("Graphviz n’est pas disponible pour générer un JPEG. Installez Graphviz (binaire 'dot').")
                    st.code(dot, language="dot")

                with st.expander("Voir la phrase complète"):
                    st.write(sel["phrase"])
                st.markdown("---")

        st.subheader("Conditions IF détectées")
        seg_if = extraire_segments_if(texte_source)
        if not seg_if:
            st.info("Aucune condition « si … » détectée.")
        else:
            for j, sel in enumerate(seg_if, start=1):
                st.markdown(f"**IF — phrase {sel['id_phrase']}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("Condition (if)")
                    st.write(sel["condition"] if sel["condition"] else "(non extraite)")
                with col2:
                    st.markdown("Action si Vrai (then)")
                    st.write(sel["action_true"] if sel["action_true"] else "(implicite ou non extraite)")
                with col3:
                    st.markdown("Action si Faux (else)")
                    st.write(sel["action_false"] if sel["action_false"] else "(absente)")

                dot = graphviz_if_dot(sel["condition"], sel["action_true"], sel["action_false"])

                if GV_OK:
                    try:
                        img_bytes = rendre_jpeg_depuis_dot(dot)
                        st.image(img_bytes, caption=f"Graphe IF — phrase {sel['id_phrase']}", use_container_width=True)
                        st.download_button(f"Télécharger le graphe IF #{sel['id_phrase']} (JPEG)",
                                           data=img_bytes,
                                           file_name=f"if_phrase_{sel['id_phrase']}.jpg",
                                           mime="image/jpeg",
                                           key=f"dl_if_jpg_{sel['id_phrase']}_{j}")
                    except Exception as e:
                        st.error(f"Rendu JPEG indisponible : {e}")
                        st.code(dot, language="dot")
                else:
                    st.error("Graphviz n’est pas disponible pour générer un JPEG. Installez Graphviz (binaire 'dot').")
                    st.code(dot, language="dot")

                with st.expander("Voir la phrase complète"):
                    st.write(sel["phrase"])
                st.markdown("---")
