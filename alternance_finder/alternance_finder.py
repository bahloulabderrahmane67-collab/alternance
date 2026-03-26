#!/usr/bin/env python3
"""
Script de correspondance d'offres d'alternance.

Ce programme charge un ensemble d'offres d'alternance à partir d'un fichier CSV,
vectorise les compétences et calcule un score de correspondance entre les
compétences de l'utilisateur et celles requises par chaque offre.

Usage :
    python3 alternance_finder.py

L'utilisateur est invité à entrer ses compétences séparées par des virgules. Le
programme affiche ensuite les offres triées par score de compatibilité.

Ce script nécessite les bibliothèques `pandas` et `numpy`.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Charge le jeu de données des offres d'alternance."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {csv_file}")
    df = pd.read_csv(csv_file)
    return df


def get_unique_skills(df: pd.DataFrame) -> list[str]:
    """Retourne la liste triée des compétences uniques dans le DataFrame."""
    skills_set: set[str] = set()
    for skills in df['Skills']:
        if pd.isna(skills):
            continue
        for skill in str(skills).split(','):
            skills_set.add(skill.strip().lower())
    return sorted(skills_set)


def build_skill_matrix(df: pd.DataFrame, all_skills: list[str]) -> np.ndarray:
    """
    Construit une matrice binaire (n_offres x n_competences).

    Chaque ligne représente une offre et chaque colonne une compétence. Une valeur de 1
    indique que l'offre requiert la compétence correspondante.
    """
    n_offres = len(df)
    n_comp = len(all_skills)
    matrix = np.zeros((n_offres, n_comp), dtype=int)
    for i, skills in enumerate(df['Skills']):
        if pd.isna(skills):
            continue
        for skill in str(skills).split(','):
            s = skill.strip().lower()
            if s in all_skills:
                j = all_skills.index(s)
                matrix[i, j] = 1
    return matrix


def vectorize_user_skills(user_input: str, all_skills: list[str]) -> np.ndarray:
    """
    Transforme les compétences saisies par l'utilisateur en vecteur binaire.

    Le vecteur contient un 1 pour chaque compétence que l'utilisateur possède et
    qui est présente dans `all_skills`.
    """
    v = np.zeros(len(all_skills), dtype=int)
    for skill in user_input.split(','):
        s = skill.strip().lower()
        if s in all_skills:
            idx = all_skills.index(s)
            v[idx] = 1
    return v


def rank_offers(df: pd.DataFrame, skill_matrix: np.ndarray, user_vector: np.ndarray) -> pd.DataFrame:
    """
    Calcule un score de correspondance pour chaque offre et retourne un DataFrame trié.

    Le score est calculé comme le produit scalaire (nombre de compétences communes).
    """
    scores = skill_matrix @ user_vector
    df_result = df.copy().reset_index(drop=True)
    df_result['match_score'] = scores
    # Trier par ordre décroissant de score, puis par salaire (optionnel) et par nom
    return df_result.sort_values(by=['match_score', 'Salary'], ascending=[False, False])


def main() -> None:
    # Chemin relatif du fichier CSV
    csv_path = Path(__file__).parent / 'data' / 'offres.csv'
    try:
        df = load_dataset(csv_path)
    except FileNotFoundError as e:
        print(e)
        return

    all_skills = get_unique_skills(df)
    if not all_skills:
        print("Aucune compétence trouvée dans le jeu de données.")
        return
    skill_matrix = build_skill_matrix(df, all_skills)

    print("\nCompétences disponibles dans la base :")
    print(', '.join(all_skills))
    user_input = input("\nEntrez vos compétences (séparées par des virgules) : ")
    user_vector = vectorize_user_skills(user_input, all_skills)
    ranked_df = rank_offers(df, skill_matrix, user_vector)
    print("\nOffres triées par score de correspondance :")
    print(ranked_df[['Company', 'Role', 'Location', 'Salary', 'match_score']].to_string(index=False))


if __name__ == '__main__':
    main()
