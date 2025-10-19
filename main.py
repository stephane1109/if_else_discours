# -*- coding: utf-8 -*-
# main.py — Discours → Code (IF / ELSE / WHILE / AND / OR) + Marqueurs + Causes/Conséquences
# Méthodes comparées : Regex vs spaCy (transformer si disponible)
#
# Fichiers requis à la racine (même dossier que ce script) :
#   - dic_code.json          : mapping des connecteurs → familles Python (IF/ELSE/WHILE/AND/OR)
#   - dict_marqueurs.json    : marqueurs normatifs (OBLIGATION/INTERDICTION/…)
#   - consequences.json      : déclencheurs de conséquence → "CONSEQUENCE"
#   - causes.json            : déclencheurs de cause → "CAUSE"
#
# Remarques :
#   - L’extraction CAUSE→CONSEQUENCE spaCy exploite la dépendance/les ancres causales et consécutives.
#   - Négation « ne … pouvoir … (pas/plus/jamais) » : ajustement par regex (sans options supplémentaires).
#   - Graphes IF/WHILE : rendu DOT (à l’écran) + export JPEG si Graphviz est présent (binaire 'dot').

import os
import re
import json
import html
import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple, Any

# =========================
# Détection Graphviz (pour export JPEG)
# =========================
try:
    import graphviz
    GV_OK = True
except Exception:
    GV_OK = False

def rendre_jpeg_depuis_dot(dot_str: str) -> bytes:
    """Rend en JPEG via graphviz.Source.pipe(format='jpg')."""
    if not GV_OK:
        raise RuntimeError("Graphviz (binaire 'dot') indisponible sur ce système.")
    src = graphviz.Source(dot_str)
    return src.pipe(format="jpg")

# =========================
# Chargement spaCy (transformer si possible)
# =========================
SPACY_OK = False
NLP = None
SPACY_STATUS: List[str] = []
try:
    import spacy
    from spacy.cli import download as spacy_download

    def _charger_modele_spacy(nom_modele: str) -> Any:
        """Tente de charger (et télécharger au besoin) un modèle spaCy FR."""
        try:
            return spacy.load(nom_modele)
        except OSError as err:
            # Modèle non installé → tentative de téléchargement automatique
            try:
                spacy_download(nom_modele)
                return spacy.load(nom_modele)
            except Exception as dl_err:
                SPACY_STATUS.append(
                    f"Téléchargement du modèle spaCy '{nom_modele}' impossible : {dl_err}"
                )
                SPACY_STATUS.append(
                    f"Erreur initiale lors du chargement de '{nom_modele}' : {err}"
                )
        except Exception as err:
            SPACY_STATUS.append(
                f"Chargement du modèle spaCy '{nom_modele}' impossible : {err}"
            )
        return None

    for name in ("fr_dep_news_trf", "fr_core_news_trf", "fr_core_news_md"):
        modele = _charger_modele_spacy(name)
        if modele is not None:
            NLP = modele
            SPACY_OK = True
            SPACY_STATUS.append(f"Modèle spaCy chargé : {name}")
            break
    if not SPACY_OK:
        SPACY_STATUS.append("Aucun modèle spaCy FR n'a pu être chargé.")
except Exception as err:
    SPACY_OK = False
    NLP = None
    SPACY_STATUS.append(f"Import de spaCy impossible : {err}")

# =========================
# Palettes / libellés Python (affichage)
# =========================
CODE_VERS_PYTHON: Dict[str, str] = {
    "IF": "if", "ELSE": "else", "WHILE": "while", "AND": "and", "OR": "or"
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
    "CONSEQUENCE": {"bg": "#fff0f0", "fg": "#b00020", "bd": "#b00020"},
    "CAUSE": {"bg": "#f0fff4", "fg": "#2f855a", "bd": "#2f855a"},
}

FAMILLES_CONNECTEURS = {"IF", "ELSE", "WHILE", "AND", "OR"}
FAMILLES_MARQUEURS_STANDARD = {
    "OBLIGATION", "INTERDICTION", "PERMISSION", "RECOMMANDATION", "SANCTION", "CADRE_OUVERTURE", "CADRE_FERMETURE"
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
# Chargement JSON à la racine
# =========================
def charger_json_dico(chemin: str) -> Dict[str, str]:
    """Charge un JSON dict { expression: etiquette } ; normalise les clés côté détection."""
    if not os.path.isfile(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    with open(chemin, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Format JSON non supporté (attendu dict) : {chemin}")
    return {normaliser_espace(k.lower()): str(v).upper() for k, v in data.items() if k and str(k).strip()}

def charger_tous_les_dicos() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Charge dic_code.json, dict_marqueurs.json, consequences.json, causes.json à la racine du projet."""
    cwd = os.getcwd()
    d_code = charger_json_dico(os.path.join(cwd, "dic_code.json"))
    d_marq = charger_json_dico(os.path.join(cwd, "dict_marqueurs.json"))
    d_cons = charger_json_dico(os.path.join(cwd, "consequences.json"))
    d_caus = charger_json_dico(os.path.join(cwd, "causes.json"))

    codes_utilises = set(d_code.values())
    attendus = FAMILLES_CONNECTEURS
    if not codes_utilises.issubset(attendus):
        raise ValueError(f"dic_code.json contient des familles inconnues : {sorted(list(codes_utilises - attendus))}")
    return d_code, d_marq, d_cons, d_caus

# =========================
# I/O discours
# =========================
def lire_fichier_txt(uploaded_file) -> str:
    """Lit un fichier .txt avec stratégie automatique d’encodage."""
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
# Détection (connecteurs / marqueurs / conséquence / cause)
# =========================
def detecter_par_dico(texte: str, dico: Dict[str, str], champ_cle: str, champ_cat: str) -> pd.DataFrame:
    """Détection générique par dictionnaire clé→étiquette (insensible à la casse)."""
    if not dico:
        return pd.DataFrame()
    texte_norm = normaliser_espace(texte)
    phrases = segmenter_en_phrases(texte_norm)
    motifs = construire_regex_depuis_liste(list(dico.keys()))
    enregs = []
    for i, ph in enumerate(phrases, start=1):
        ph_norm = normaliser_espace(ph)
        for cle_norm, motif in motifs:
            for m in motif.finditer(ph_norm):
                enregs.append({
                    "id_phrase": i,
                    "phrase": ph.strip(),
                    champ_cle: cle_norm,
                    champ_cat: dico[cle_norm],
                    "position": m.start(),
                    "longueur": m.end() - m.start()
                })
    df = pd.DataFrame(enregs)
    if not df.empty:
        df.sort_values(by=["id_phrase", "position"], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)
    return df

def detecter_connecteurs(texte: str, dico_conn: Dict[str, str]) -> pd.DataFrame:
    return detecter_par_dico(texte, dico_conn, "connecteur", "code")

def detecter_marqueurs(texte: str, dico_marq: Dict[str, str]) -> pd.DataFrame:
    return detecter_par_dico(texte, dico_marq, "marqueur", "categorie")

def detecter_consequences_lex(texte: str, dico_consq: Dict[str, str]) -> pd.DataFrame:
    # On ajoute une colonne 'consequence' pour homogénéiser les affichages regex
    df = detecter_par_dico(texte, dico_consq, "consequence", "categorie")
    return df

def detecter_causes_lex(texte: str, dico_causes: Dict[str, str]) -> pd.DataFrame:
    # On ajoute une colonne 'cause' pour homogénéiser les affichages regex
    df = detecter_par_dico(texte, dico_causes, "cause", "categorie")
    return df

# =========================
# Négation (regex autour de "pouvoir")
# =========================
def ajuster_negations_regex(phrase: str, dets_phrase: pd.DataFrame) -> pd.DataFrame:
    """
    Reclasse en INTERDICTION les formes 'ne … peut/peuvent/pourra/pourront (pas/plus/jamais)'.
    Ne traite pas d'autre verbe ; pas d'option 'ne … peut' sans 'pas'.
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

def ajuster_negations_global(texte: str, df_marq: pd.DataFrame) -> pd.DataFrame:
    """Applique les ajustements de négation phrase par phrase (regex)."""
    if df_marq.empty:
        return df_marq
    phrases = segmenter_en_phrases(texte)
    dets_list = []
    for i, ph in enumerate(phrases, start=1):
        bloc = df_marq[df_marq["id_phrase"] == i].copy()
        if bloc.empty:
            dets_list.append(bloc); continue
        bloc = ajuster_negations_regex(ph, bloc)
        dets_list.append(bloc)
    df_adj = pd.concat(dets_list, ignore_index=True) if dets_list else df_marq
    if not df_adj.empty:
        df_adj.sort_values(by=["id_phrase","position"], inplace=True, kind="mergesort")
        df_adj.reset_index(drop=True, inplace=True)
    return df_adj

# =========================
# Annotation HTML (texte + badges)
# =========================
def _esc(s: str) -> str:
    return html.escape(s, quote=False)

def css_badges() -> str:
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
                      dico_marq: Dict[str, str],
                      dico_consq: Dict[str, str],
                      dico_causes: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fusionne toutes les occurrences, élimine les chevauchements (priorité au plus long)."""
    occs: List[Dict[str, Any]] = []
    for typ, dico in [
        ("connecteur", dico_conn),
        ("marqueur", dico_marq),
        ("consequence", dico_consq),
        ("cause", dico_causes),
    ]:
        if not dico:
            continue
        motifs = construire_regex_depuis_liste(list(dico.keys()))
        for cle, motif in motifs:
            for m in motif.finditer(texte):
                occs.append({
                    "debut": m.start(), "fin": m.end(), "original": texte[m.start():m.end()],
                    "type": typ, "cle": cle, "etiquette": dico[cle], "longueur": m.end()-m.start(),
                })
    occs.sort(key=lambda x: (x["debut"], -x["longueur"]))
    res = []; borne = -1
    for m in occs:
        if m["debut"] >= borne:
            res.append(m); borne = m["fin"]
    return res

def libelle_python(code: str) -> str:
    return CODE_VERS_PYTHON.get(str(code).upper(), str(code).upper())

def html_annote(texte: str,
                dico_conn: Dict[str, str],
                dico_marq: Dict[str, str],
                dico_consq: Dict[str, str],
                dico_causes: Dict[str, str],
                show_codes: Dict[str, bool],
                show_marqueurs: bool,
                show_consequences: bool,
                show_causes: bool,
                filtres_marqueurs: List[str] = None) -> str:
    """Produit le HTML annoté selon les cases cochées."""
    if not texte:
        return "<div class='texte-annote'>(Texte vide)</div>"
    filtres_marqueurs = set([c.upper() for c in (filtres_marqueurs or [])])

    t = texte
    occ = occurrences_mixte(t, dico_conn, dico_marq, dico_consq, dico_causes)
    morceaux: List[str] = []; curseur = 0
    for m in occ:
        # Filtres d’affichage
        if m["type"] == "connecteur":
            code = str(m["etiquette"]).upper()
            if not show_codes.get(code, True):
                continue
        elif m["type"] == "marqueur":
            if not show_marqueurs:
                continue
            cat = str(m["etiquette"]).upper()
            if filtres_marqueurs and cat not in filtres_marqueurs:
                continue
        elif m["type"] == "consequence":
            if not show_consequences:
                continue
        elif m["type"] == "cause":
            if not show_causes:
                continue

        if m["debut"] > curseur:
            morceaux.append(_esc(t[curseur:m["debut"]]))

        mot_original = _esc(t[m["debut"]:m["fin"]])
        if m["type"] == "connecteur":
            code = str(m["etiquette"]).upper()
            badge = f"<span class='badge-code code-{_esc(code)}'>{_esc(libelle_python(code))}</span>"
            rendu = f"<span class='connecteur'>{mot_original}</span>{badge}"
        else:
            cat_disp = str(m["etiquette"]).upper()
            badge = f"<span class='badge-marqueur marq-{_esc(cat_disp)}'>{_esc(cat_disp)}</span>"
            rendu = f"<span class='mot-marque'>{mot_original}</span>{badge}"

        morceaux.append(rendu); curseur = m["fin"]

    if curseur < len(t):
        morceaux.append(_esc(t[curseur:]))

    if not morceaux:
        return "<div class='texte-annote'>(Aucune annotation selon les cases sélectionnées)</div>"
    return "<div class='texte-annote'>" + "".join(morceaux) + "</div>"

def html_autonome(fragment_html: str) -> str:
    return f"<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'/><title>Texte annoté</title>{css_badges()}</head><body>{fragment_html}</body></html>"

# =========================
# Extraction graphes WHILE / IF (heuristiques)
# =========================
def extraire_segments_while(texte: str) -> List[Dict[str, Any]]:
    """Extrait 'tant que ...' ; condition = segment après 'tant que' jusqu’à la 1re ponctuation, action = suite éventuelle."""
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

def _proposition_principale_apres_dernier_si(phrase: str, last_si_end: int, df_consq_phrase: pd.DataFrame) -> str:
    """Si un déclencheur de conséquence apparaît après la dernière condition, prendre ce qui suit ; sinon couper à la 1re ponctuation."""
    reste = phrase[last_si_end:].strip()
    if df_consq_phrase is not None and not df_consq_phrase.empty:
        locs = sorted(set(df_consq_phrase["consequence"].tolist()), key=lambda s: len(s), reverse=True)
        for loc in locs:
            pat = re.compile(rf"(?<![A-Za-zÀ-ÖØ-öø-ÿ]){re.escape(loc)}(?![A-Za-zÀ-ÖØ-öø-ÿ])", flags=re.I)
            m = pat.search(phrase, pos=last_si_end)
            if m:
                return phrase[m.end():].strip(" .;:-—")
    m = re.search(r"[,;:\-\—]\s+", reste)
    if m:
        return reste[m.end():].strip(" .;:-—")
    return reste.strip(" .;:-—")

def extraire_segments_if(texte: str, df_consq_global: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extrait toutes les occurrences 'si ...' ; action si vrai = apodose commune après la dernière condition ; else si 'sinon' présent."""
    res = []
    phrases = segmenter_en_phrases(texte)
    for idx, ph in enumerate(phrases, start=1):
        matches = list(re.finditer(r"\bsi\b|\bs['’]il(?:s)?\b|\bsi\s+l['’]on\b|\bsi\s+on\b", ph, flags=re.I))
        if not matches:
            continue

        action_false = ""
        m_sinon = re.search(r"\bsinon\b", ph, flags=re.I)
        if m_sinon:
            action_false = ph[m_sinon.end():].strip(" .;:-—")

        cons_this = pd.DataFrame()
        if df_consq_global is not None and not df_consq_global.empty:
            cons_this = df_consq_global[df_consq_global["id_phrase"] == idx]

        action_true_gauche = ph[:matches[0].start()].strip(" .;:-—")

        def fin_clause_si(start_pos: int) -> int:
            sub = ph[start_pos:].strip()
            m = re.search(r"[,;:\-\—]\s+", sub)
            if m:
                return start_pos + m.start()
            return len(ph)

        last = matches[-1]
        last_clause_end = fin_clause_si(last.end())
        action_true_droite = _proposition_principale_apres_dernier_si(ph, last_clause_end, cons_this)
        action_true_commune = action_true_gauche if action_true_gauche else action_true_droite

        for k, m in enumerate(matches):
            debut_cond = m.end()
            segment = ph[debut_cond:]
            m_ponc = re.search(r"[,;:\-\—]\s+", segment)
            fin_cond_rel = m_ponc.start() if m_ponc else len(segment)
            if k + 1 < len(matches):
                fin_cond_rel = min(fin_cond_rel, matches[k+1].start() - debut_cond)
            condition = segment[:fin_cond_rel].strip()

            res.append({
                "type": "IF",
                "id_phrase": idx,
                "phrase": ph.strip(),
                "condition": condition,
                "action_true": action_true_commune,
                "action_false": action_false.strip(),
                "debut": m.start(),
            })
    return res

def graphviz_while_dot(condition: str, action: str) -> str:
    """Construit un DOT simple pour WHILE."""
    def esc(s: str) -> str: return s.replace('"', r"\"")
    cond_txt = esc(condition if condition else "(condition non extraite)")
    act_txt = esc(action if action else "(action implicite ou non extraite)")
    return f'''
digraph G {{
  rankdir=LR;
  node [shape=box, fontname="Helvetica"];
  start [shape=circle, label="Start"];
  cond [shape=diamond, label="while ({cond_txt})"];
  act  [shape=box, label="{act_txt}"];
  end  [shape=doublecircle, label="End"];
  start -> cond;
  cond -> act [label="Vrai"];
  act  -> cond [label="itère"];
  cond -> end [label="Faux"];
}}
'''

def graphviz_if_dot(condition: str, action_true: str, action_false: str = "") -> str:
    """Construit un DOT simple pour IF."""
    def esc(s: str) -> str: return s.replace('"', r"\"")
    cond_txt = esc(condition if condition else "(condition non extraite)")
    act_t = esc(action_true if action_true else "(action si vrai non extraite)")
    act_f = esc(action_false) if action_false else ""
    has_else = bool(action_false.strip())
    lignes = [
        "digraph G {",
        "  rankdir=LR;",
        '  node [shape=box, fontname="Helvetica"];',
        '  start [shape=circle, label="Start"];',
        f'  cond  [shape=diamond, label="if ({cond_txt})"];',
        f'  actt  [shape=box, label="Action si Vrai: {act_t}"];',
    ]
    if has_else:
        lignes.append(f'  actf  [shape=box, label="Action si Faux: {act_f}"];')
    lignes.append('  end   [shape=doublecircle, label="End"];')
    lignes += [
        "  start -> cond;",
        '  cond  -> actt [label="Vrai"];',
        "  actt  -> end;",
    ]
    if has_else:
        lignes += ['  cond  -> actf [label="Faux"];', "  actf  -> end;"]
    lignes.append("}")
    return "\n".join(lignes)

# =========================
# spaCy : extraction CAUSE → CONSÉQUENCE
# =========================
def _locution_match(tok, locutions_norm: set) -> bool:
    """Teste un match simple sur le token lui-même ou sa sous-chaîne de sous-arbre."""
    t = tok.lower_
    if t in locutions_norm:
        return True
    surface = " ".join(w.lower_ for w in tok.subtree)
    return any(loc in surface for loc in locutions_norm)

def extraire_cause_consequence_spacy(texte: str, nlp, causes_lex: List[str], consequences_lex: List[str]) -> pd.DataFrame:
    """
    Retourne un DataFrame avec les segments CAUSE/CONSÉQUENCE extraits par analyse dépendancielle.
    Colonnes : id_phrase, type, cause_span, consequence_span, ancre, methode, phrase
    """
    if not nlp:
        return pd.DataFrame()

    doc = nlp(texte)
    causes_norm = {c.lower().strip() for c in causes_lex}
    consq_norm = {c.lower().strip() for c in consequences_lex}
    enregs = []
    prev_sent_text = ""

    for pid, sent in enumerate(doc.sents, start=1):
        # Subordonnées causales (mark ∈ causes)
        for tok in sent:
            if tok.dep_.lower() == "mark" and _locution_match(tok, causes_norm):
                head = tok.head
                cause_span = doc[head.left_edge.i: head.right_edge.i+1]
                enregs.append({
                    "id_phrase": pid,
                    "type": "CAUSE_SUBORDONNEE",
                    "cause_span": cause_span.text,
                    "consequence_span": sent.text,
                    "ancre": tok.text,
                    "methode": "mark→advcl",
                    "phrase": sent.text
                })

        # Groupes prépositionnels causaux (à cause de, en raison de, du fait de, grâce à…)
        for tok in sent:
            if tok.dep_.lower() in {"case","fixed","mark"} and _locution_match(tok, causes_norm):
                head = tok.head
                gn = doc[head.left_edge.i: head.right_edge.i+1]
                enregs.append({
                    "id_phrase": pid,
                    "type": "CAUSE_GN",
                    "cause_span": gn.text,
                    "consequence_span": sent.text,
                    "ancre": tok.text,
                    "methode": "case/fixed→obl",
                    "phrase": sent.text
                })

        # Conséquence adverbiale en tête de phrase (donc, alors, ainsi, dès lors…)
        premiers = [t for t in sent if not t.is_punct][:3]
        if premiers:
            t0 = premiers[0]
            if _locution_match(t0, consq_norm) and t0.pos_ in {"ADV","CCONJ","SCONJ"}:
                enregs.append({
                    "id_phrase": pid,
                    "type": "CONSEQUENCE_ADV",
                    "cause_span": prev_sent_text,
                    "consequence_span": sent.text,
                    "ancre": t0.text,
                    "methode": "adv/discourse tête de phrase",
                    "phrase": sent.text
                })

        # Subordonnées consécutives (de sorte que, si bien que, de façon que…)
        for tok in sent:
            if tok.dep_.lower() == "mark" and _locution_match(tok, consq_norm):
                head = tok.head
                cons_span = doc[head.left_edge.i: head.right_edge.i+1]
                enregs.append({
                    "id_phrase": pid,
                    "type": "CONSEQUENCE_SUBORDONNEE",
                    "cause_span": sent.text,
                    "consequence_span": cons_span.text,
                    "ancre": tok.text,
                    "methode": "mark→advcl(consécutif)",
                    "phrase": sent.text
                })

        prev_sent_text = sent.text

    df = pd.DataFrame(enregs)
    if not df.empty:
        df = df.sort_values(["id_phrase", "type"]).reset_index(drop=True)
    return df

# =========================
# Helpers pour tableaux comparatifs (surlignage ⟦ … ⟧)
# =========================
def marquer_terme_brut(phrase: str, terme: str) -> str:
    """Entoure la 1ère occurrence de 'terme' par ⟦…⟧ (insensible à la casse), sans HTML."""
    if not phrase or not terme:
        return phrase or ""
    m = re.search(re.escape(terme), phrase, flags=re.I)
    if not m:
        return phrase
    i, j = m.start(), m.end()
    return phrase[:i] + "⟦" + phrase[i:j] + "⟧" + phrase[j:]

def table_regex_df(df: pd.DataFrame, type_marqueur: str) -> pd.DataFrame:
    """
    Construit un DataFrame pour l’affichage Streamlit :
      - type_marqueur = "CAUSE"  -> colonne clé = 'cause'
      - type_marqueur = "CONSEQUENCE" -> colonne clé = 'consequence'
    Ajoute 'phrase_marquee' avec le marqueur entouré de ⟦…⟧.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["id_phrase", "marqueur", "catégorie", "phrase_marquee"])

    cle = "cause" if type_marqueur.upper() == "CAUSE" else "consequence"
    lignes = []
    for _, row in df.iterrows():
        marqueur = row.get(cle, "")
        cat = row.get("categorie", "")
        phr = row.get("phrase", "")
        phr_m = marquer_terme_brut(phr, marqueur)
        lignes.append({
            "id_phrase": row.get("id_phrase", ""),
            "marqueur": marqueur,
            "catégorie": cat,
            "phrase_marquee": phr_m
        })
    return pd.DataFrame(lignes)

def table_spacy_df(df_spacy: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un DataFrame pour l’affichage Streamlit côté spaCy :
    Colonnes : id_phrase, type, ancre, méthode, phrase_marquee, cause_span, consequence_span
    (l’ancre est entourée de ⟦…⟧ dans la phrase).
    """
    if df_spacy is None or df_spacy.empty:
        return pd.DataFrame(columns=["id_phrase", "type", "ancre", "méthode", "phrase_marquee", "cause_span", "consequence_span"])

    lignes = []
    for _, row in df_spacy.iterrows():
        ancre = row.get("ancre", "")
        phr = row.get("phrase", "")
        phr_m = marquer_terme_brut(phr, ancre)
        lignes.append({
            "id_phrase": row.get("id_phrase", ""),
            "type": row.get("type", ""),
            "ancre": ancre,
            "méthode": row.get("methode", ""),
            "phrase_marquee": phr_m,
            "cause_span": row.get("cause_span", ""),
            "consequence_span": row.get("consequence_span", "")
        })
    return pd.DataFrame(lignes)

# =========================
# Interface Streamlit
# =========================
st.set_page_config(page_title="Discours → Code (Regex vs spaCy + JSON racine)", page_icon=None, layout="wide")
st.title("Discours → Code : IF / ELSE / WHILE / AND / OR + marqueurs + causes/conséquences (Regex vs spaCy)")

# Chargement des dicos
try:
    DICO_CONNECTEURS, DICO_MARQUEURS, DICO_CONSQS, DICO_CAUSES = charger_tous_les_dicos()
except Exception as e:
    st.error("Impossible de charger les dictionnaires JSON à la racine.")
    st.code(str(e))
    st.stop()

# Alerte spaCy/Graphviz
if not SPACY_OK:
    st.warning(
        "spaCy FR indisponible (transformer préféré, fallback md si disponible). L’onglet spaCy utilisera uniquement Regex si aucun modèle FR n’est chargé."
    )
if SPACY_STATUS:
    st.caption(" ; ".join(SPACY_STATUS))
if not GV_OK:
    st.warning("Graphviz non détecté : l’export JPEG des graphes ne sera pas disponible (rendu DOT affiché quand même).")

# Barre latérale : choix méthodes
with st.sidebar:
    st.header("Méthodes d’analyse")
    use_regex_cc = st.checkbox("Causalité par Regex (dictionnaires JSON)", value=True)
    use_spacy_cc = st.checkbox("Causalité par spaCy (analyse NLP)", value=SPACY_OK)

# Source du discours
st.markdown("### Source du discours")
mode_source = st.radio("Choisir la source du texte", ["Fichier .txt", "Zone de texte"], index=0, horizontal=True)

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

# Détections de base
if texte_source.strip():
    df_conn = detecter_connecteurs(texte_source, DICO_CONNECTEURS)
    df_marq_brut = detecter_marqueurs(texte_source, DICO_MARQUEURS)
    df_marq = ajuster_negations_global(texte_source, df_marq_brut)
    df_consq_lex = detecter_consequences_lex(texte_source, DICO_CONSQS) if use_regex_cc else pd.DataFrame()
    df_causes_lex = detecter_causes_lex(texte_source, DICO_CAUSES) if use_regex_cc else pd.DataFrame()
else:
    df_conn = pd.DataFrame()
    df_marq = pd.DataFrame()
    df_consq_lex = pd.DataFrame()
    df_causes_lex = pd.DataFrame()

# Onglets
ong1, ong2, ong3, ong4, ong5, ong6 = st.tabs([
    "Expressions mappées", "Détections", "Dictionnaires (JSON)", "Guide d’interprétation", "Graphiques (IF / WHILE)", "Comparatif Regex / spaCy"
])

# Onglet 1 : Expressions mappées
with ong1:
    st.subheader("Expressions françaises mappées vers une famille Python (if / else / while / and / or)")
    if not DICO_CONNECTEURS:
        st.info("Aucun connecteur chargé.")
    else:
        df_map = pd.DataFrame(sorted([(k, v) for k, v in DICO_CONNECTEURS.items()], key=lambda x: (x[1], x[0])),
                              columns=["expression_française", "famille_python"])
        st.dataframe(df_map, use_container_width=True, hide_index=True)
        st.download_button("Exporter le mappage (CSV)",
                           data=df_map.to_csv(index=False).encode("utf-8"),
                           file_name="mapping_connecteurs.csv", mime="text/csv",
                           key="dl_map_connecteurs_csv")

# Onglet 2 : Détections (listes + texte annoté)
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

    st.subheader("Marqueurs détectés")
    if df_marq.empty:
        st.info("Aucun marqueur détecté.")
    else:
        st.dataframe(df_marq, use_container_width=True, hide_index=True)
        st.download_button("Exporter marqueurs (CSV)",
                           data=df_marq.to_csv(index=False).encode("utf-8"),
                           file_name="occurrences_marqueurs.csv", mime="text/csv",
                           key="dl_occ_marq_csv")

    colX, colY = st.columns(2)
    with colX:
        st.subheader("Déclencheurs de conséquence (Regex)")
        if not use_regex_cc:
            st.info("Méthode Regex désactivée (voir barre latérale).")
        else:
            if df_consq_lex.empty:
                st.info("Aucun déclencheur de conséquence détecté par Regex.")
            else:
                st.dataframe(df_consq_lex, use_container_width=True, hide_index=True)
                st.download_button("Exporter conséquences (CSV)",
                                   data=df_consq_lex.to_csv(index=False).encode("utf-8"),
                                   file_name="occurrences_consequences.csv", mime="text/csv",
                                   key="dl_occ_consq_csv")
    with colY:
        st.subheader("Déclencheurs de cause (Regex)")
        if not use_regex_cc:
            st.info("Méthode Regex désactivée (voir barre latérale).")
        else:
            if df_causes_lex.empty:
                st.info("Aucun déclencheur de cause détecté par Regex.")
            else:
                st.dataframe(df_causes_lex, use_container_width=True, hide_index=True)
                st.download_button("Exporter causes (CSV)",
                                   data=df_causes_lex.to_csv(index=False).encode("utf-8"),
                                   file_name="occurrences_causes.csv", mime="text/csv",
                                   key="dl_occ_causes_csv")

    st.markdown("---")
    st.subheader("Texte annoté (filtres par familles)")

    # Cases à cocher pour les familles de connecteurs et marqueurs
    colA, colB, colC, colD, colE, colF, colG, colH = st.columns(8)
    with colA:
        show_if = st.checkbox("IF (si)", value=True)
    with colB:
        show_else = st.checkbox("ELSE (sinon)", value=True)
    with colC:
        show_while = st.checkbox("WHILE (tant que)", value=True)
    with colD:
        show_and = st.checkbox("AND (et)", value=True)
    with colE:
        show_or = st.checkbox("OR (ou)", value=True)
    with colF:
        show_marqueurs = st.checkbox("Marqueurs (hors CAUSE/CONSEQUENCE)", value=True)
    with colG:
        show_consequences = st.checkbox("CONSEQUENCE", value=True)
    with colH:
        show_causes = st.checkbox("CAUSE", value=True)

    categories_disponibles = sorted(list(FAMILLES_MARQUEURS_STANDARD))
    filt_marq = st.multiselect("Limiter aux catégories de marqueurs (optionnel)", categories_disponibles, default=[])

    show_codes = {"IF": show_if, "ELSE": show_else, "WHILE": show_while, "AND": show_and, "OR": show_or}

    st.markdown(css_badges(), unsafe_allow_html=True)
    if not texte_source.strip():
        st.info("Aucun texte fourni.")
        frag = "<div class='texte-annote'>(Texte vide)</div>"
    else:
        frag = html_annote(
            texte_source,
            DICO_CONNECTEURS,
            DICO_MARQUEURS,
            DICO_CONSQS if show_consequences else {},
            DICO_CAUSES if show_causes else {},
            show_codes=show_codes,
            show_marqueurs=show_marqueurs,
            show_consequences=show_consequences,
            show_causes=show_causes,
            filtres_marqueurs=filt_marq
        )
        st.markdown(frag, unsafe_allow_html=True)
        st.download_button("Exporter le texte annoté (HTML)",
                           data=html_autonome(frag).encode("utf-8"),
                           file_name="texte_annote.html", mime="text/html",
                           key="dl_annote_html")

# Onglet 3 : Dictionnaires (JSON)
with ong3:
    st.subheader("Aperçu des dictionnaires chargés (racine)")
    st.markdown("**dic_code.json**")
    st.json(DICO_CONNECTEURS, expanded=False)
    st.markdown("**dict_marqueurs.json**")
    st.json(DICO_MARQUEURS, expanded=False)
    st.markdown("**consequences.json**")
    st.json(DICO_CONSQS, expanded=False)
    st.markdown("**causes.json**")
    st.json(DICO_CAUSES, expanded=False)

# Onglet 4 : Guide d’interprétation
with ong4:
    st.subheader("Guide d’interprétation : Python vs analyse de discours")

    st.markdown("#### IF")
    st.write(
        "Cadre Python : `if` introduit une condition booléenne ; le bloc indenté s’exécute si la condition est vraie. "
        "On peut enchaîner `elif` puis `else` ; la condition évalue `True/False` (valeurs truthy/falsy admises).\n\n"
        "Cadre analyse : « si », « à condition que », « pourvu que », « au cas où » posent une condition d’acceptabilité ou de mise en œuvre."
    )

    st.markdown("#### ELSE")
    st.write(
        "Cadre Python : `else` est la branche alternative quand la condition est fausse.\n\n"
        "Cadre analyse : « sinon », « autrement », « à défaut » marquent l’alternative par défaut ou le coût si la condition n’est pas remplie."
    )

    st.markdown("#### WHILE")
    st.write(
        "Cadre Python : `while` répète un bloc tant que l’expression reste vraie ; `break` sort ; `continue` saute l’itération suivante.\n\n"
        "Cadre analyse : « tant que » exprime une persistance conditionnelle (maintien d’une action tant qu’un état perdure)."
    )

    st.markdown("#### AND")
    st.write(
        "Cadre Python : `and` exige que toutes les sous-conditions soient vraies (court-circuit logique).\n\n"
        "Cadre analyse : « et », « ainsi que » agrègent des critères/engagements pour construire un front commun."
    )

    st.markdown("#### OR")
    st.write(
        "Cadre Python : `or` vaut vrai si au moins une sous-condition est vraie (court-circuit).\n\n"
        "Cadre analyse : « ou », « ou bien », « soit » posent des alternatives."
    )

    st.markdown("#### Déclencheurs de conséquence (CONSEQUENCE)")
    st.write(
        "Repères pour l’apodose : « donc », « alors », « c’est pourquoi », "
        "« par conséquent », « de ce fait », « ainsi », « dès lors », « en conséquence », etc."
    )

    st.markdown("#### Déclencheurs de cause (CAUSE)")
    st.write(
        "Repères qui motivent/justifient un fait : « parce que », « car », « puisque », « comme » (en tête), "
        "« en raison de », « du fait que », « à cause de », « grâce à », « faute de », « suite à », etc."
    )

# Onglet 5 : Graphiques (IF / WHILE)
with ong5:
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
                st.graphviz_chart(dot, use_container_width=True)

                if GV_OK:
                    try:
                        img_bytes = rendre_jpeg_depuis_dot(dot)
                        st.download_button(f"Télécharger le graphe WHILE #{sel['id_phrase']} (JPEG)",
                                           data=img_bytes,
                                           file_name=f"while_phrase_{sel['id_phrase']}.jpg",
                                           mime="image/jpeg",
                                           key=f"dl_while_jpg_{sel['id_phrase']}_{i}")
                    except Exception as e:
                        st.error(f"Export JPEG indisponible : {e}")

                with st.expander("Voir la phrase complète"):
                    st.write(sel["phrase"])
                st.markdown("---")

        st.subheader("Conditions IF détectées (toutes occurrences)")
        df_consq_for_if = df_consq_lex if not df_consq_lex.empty else pd.DataFrame()
        seg_if = extraire_segments_if(texte_source, df_consq_for_if)
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
                    st.markdown("Action si Vrai")
                    st.write(sel["action_true"] if sel["action_true"] else "(implicite ou non extraite)")
                with col3:
                    st.markdown("Action si Faux (else)")
                    st.write(sel["action_false"] if sel["action_false"] else "(absente)")

                dot = graphviz_if_dot(sel["condition"], sel["action_true"], sel["action_false"])
                st.graphviz_chart(dot, use_container_width=True)

                if GV_OK:
                    try:
                        img_bytes = rendre_jpeg_depuis_dot(dot)
                        st.download_button(f"Télécharger le graphe IF #{sel['id_phrase']} (JPEG)",
                                           data=img_bytes,
                                           file_name=f"if_phrase_{sel['id_phrase']}.jpg",
                                           mime="image/jpeg",
                                           key=f"dl_if_jpg_{sel['id_phrase']}_{j}")
                    except Exception as e:
                        st.error(f"Export JPEG indisponible : {e}")

                with st.expander("Voir la phrase complète"):
                    st.write(sel["phrase"])
                st.markdown("---")

# Onglet 6 : Comparatif Regex / spaCy
with ong6:
    st.subheader("Comparatif des détections Causes/Conséquences : Regex vs spaCy")

    if not texte_source.strip():
        st.info("Aucun texte fourni.")
    else:
        # 1) Regex — CAUSE
        st.markdown("**Détections Regex — CAUSE**")
        if not use_regex_cc:
            st.info("Méthode Regex désactivée (voir barre latérale).")
        else:
            if df_causes_lex.empty:
                st.info("Aucune CAUSE trouvée par Regex.")
            else:
                df_view_cause = table_regex_df(df_causes_lex, "CAUSE")
                st.dataframe(df_view_cause, use_container_width=True, hide_index=True)

        st.markdown("---")

        # 2) Regex — CONSEQUENCE
        st.markdown("**Détections Regex — CONSEQUENCE**")
        if not use_regex_cc:
            st.info("Méthode Regex désactivée (voir barre latérale).")
        else:
            if df_consq_lex.empty:
                st.info("Aucune CONSEQUENCE trouvée par Regex.")
            else:
                df_view_consq = table_regex_df(df_consq_lex, "CONSEQUENCE")
                st.dataframe(df_view_consq, use_container_width=True, hide_index=True)

        st.markdown("---")

        # 3) spaCy — CAUSE → CONSÉQUENCE
        st.markdown("**Détections spaCy — CAUSE → CONSÉQUENCE**")
        if use_spacy_cc and SPACY_OK and NLP is not None:
            df_cc_spacy = extraire_cause_consequence_spacy(
                texte_source,
                NLP,
                list(DICO_CAUSES.keys()),
                list(DICO_CONSQS.keys())
            )
            if df_cc_spacy.empty:
                st.info("Aucun lien trouvé par spaCy (selon les ancres fournies).")
            else:
                df_spacy_view = table_spacy_df(df_cc_spacy)
                st.dataframe(df_spacy_view, use_container_width=True, hide_index=True)
                st.download_button(
                    "Exporter CAUSE → CONSÉQUENCE (CSV)",
                    data=df_spacy_view.to_csv(index=False).encode("utf-8"),
                    file_name="cause_consequence_spacy.csv",
                    mime="text/csv",
                    key="dl_cc_spacy_csv"
                )
        elif use_spacy_cc and not SPACY_OK:
            st.warning("spaCy FR indisponible (installez un modèle français, idéalement transformer).")
        else:
            st.info("spaCy désactivé (voir la barre latérale).")
