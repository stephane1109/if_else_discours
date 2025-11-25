from textwrap import dedent

import streamlit as st


def render_lexique_tab() -> None:
    """Affiche le contenu de l'onglet Lexique."""
    st.markdown("#### Connecteurs logique")
    st.markdown(
        dedent(
            """Ce script réalise une analyse automatique de discours en s’appuyant sur la logique conditionnelle de la
            programmation (notamment Python). Il parcourt le corpus puis applique une série de règles de type « SI… ALORS…
            SINON… TANT QUE » afin de catégoriser le texte selon ces logiques discursives."""
        )
    )

    st.markdown("#### Marqueurs")
    st.markdown(
        dedent(
            """Le dictionnaire « Marqueurs » regroupe l’ensemble des formes linguistiques et expressions récurrentes qui
            signalent, dans le discours (politique), des prises de position normatives ou des cadrages spécifiques. Le script
            identifie automatiquement, dans le texte, les moments où l’orateur formule une obligation, une interdiction, une
            permission, une recommandation, une sanction, ou encore ouvre ou clôt un cadre discursif."""
        )
    )
    st.markdown("#### Marqueurs \"cause/conséquence\"")
    st.markdown(
        "Les dictionnaires « Causes » et « Conséquences ». Ces dictionnaires rassemblent les expressions, connecteurs et constructions syntaxiques qui signalent dans le discours des relations de causalité (ce qui produit, motive ou explique) et des relations de conséquence (ce qui résulte, découle ou est présenté comme l’effet d’une cause)."
    )
    st.markdown("#### Lexique des termes clés")
    st.markdown(
        "- **APODOSE** : Proposition principale qui, placée après une subordonnée conditionnelle (protase), en indique la conséquence. Exemple : *Si j’insiste* (protase), *il viendra* (apodose)."
    )

    st.markdown("#### Connecteurs logiques (façon Python)")
    st.markdown(
        "- **IF (si…)** : introduit une condition ; ce qui suit dépend du fait que la condition soit vraie.\n"
        "- **ELSE (sinon)** : propose l’alternative quand la condition précédente n’est pas remplie.\n"
        "- **WHILE (tant que)** : action répétée tant qu’une condition reste vraie, marque une persistance.\n"
        "- **AND (et)** : additionne ou exige plusieurs conditions/éléments à la fois.\n"
        "- **OR (ou/soit)** : offre des alternatives ; au moins une suffit."
    )

    st.markdown("#### Marqueurs normatifs (prescription)")
    st.markdown(
        "- **OBLIGATION** : ce qui doit être fait (nécessité, devoir).\n"
        "- **INTERDICTION** : ce qui ne doit pas être fait (empêchement, prohibition).\n"
        "- **PERMISSION** : ce qui est autorisé ou possible.\n"
        "- **RECOMMANDATION** : ce qu’il vaut mieux faire (conseil, souhaitable).\n"
        "- **SANCTION** : annonce une punition ou un coût en cas d’écart.\n"
        "- **CADRE_OUVERTURE** : invite à débattre ou à élargir le dialogue.\n"
        "- **CADRE_FERMETURE** : clôt ou restreint le débat (pas le moment, pas de polémique)."
    )

    st.markdown("#### Relations causales")
    st.markdown(
        "- **CAUSE** : justifie ou explique un fait (parce que, car, en raison de…).\n"
        "- **CONSEQUENCE** : en déduit l’effet ou l’issue (donc, alors, par conséquent…)."
    )

    st.markdown("#### Marqueurs \"mémoire\" (cognitifs)")
    st.markdown(
        "- **MEM_PERS** : souvenirs ou expériences personnelles mobilisés dans le discours.\n"
        "- **MEM_COLL** : mémoire partagée, appel au « nous » collectif.\n"
        "- **MEM_RAPPEL** : injonctions ou formules pour ne pas oublier un fait.\n"
        "- **MEM_RENVOI** : renvoi à des propos ou engagements déjà formulés.\n"
        "- **MEM_REPET** : marqueurs de répétition ou d’insistance (encore, déjà…).\n"
        "- **MEM_PASSE** : ancrage explicite dans un temps passé (jadis, auparavant…)."
    )

    st.markdown("#### Marqueurs \"si… alors / sinon\"")
    st.markdown(
        "- **CONDITION** : déclenche la condition (« si… », « à condition que… »).\n"
        "- **APODOSE / ALORS** : pointe la conséquence attendue (« alors », « donc », « dès lors… »)."
    )

    st.markdown("#### Lexique grammatical spaCy (causalité)")
    st.markdown(
        "- **CAUSE_GN** : cause exprimée par un groupe nominal (souvent introduit par *en raison de*, *à cause de*…). Exemple : *En raison de la tempête, le match est reporté.*\n"
        "- **CONSEQUENCE_ADV** : conséquence exprimée par un adverbe ou un groupe adverbial (ex. *donc*, *par conséquent*). Exemple : *Il a tout expliqué, par conséquent nous comprenons la décision.*\n"
        "- **CAUSE_SUBORDONNEE** : cause formulée par une proposition subordonnée (ex. *parce que*, *puisque*…). Exemple : *Parce qu’il pleuvait, la cérémonie a été déplacée à l’intérieur.*"
    )
