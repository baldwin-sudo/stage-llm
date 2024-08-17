from llm_process.score_prediction import calculate_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Charger les données des objectifs et des pourcentages
objectif_df = pd.read_csv("./objectif_pourc.csv")

def audit_strategique(user_answer, question_id):
    '''
    Évaluer la réponse de l'utilisateur en fonction des objectifs définis et retourner les scores.

    Input:
        user_answer (String): La réponse de l'utilisateur.
        question_id (int): L'ID de la question.

    Output:
        dict: Un dictionnaire avec les objectifs comme clés et les scores correspondants comme valeurs.
    '''
    objectifs = objectif_df['Objectif'].unique()
    user_score = calculate_score(user_answer, question_id,use_tfidf=False,use_lsa=False)
    score_objectifs = {objectif: 0 for objectif in objectifs}
    
    for _, row in objectif_df.iterrows():
        objectif = row['Objectif']
        pourcentage = float(row['Pourcentage'].strip('%')) / 100
        score_objectifs[objectif] = user_score * pourcentage
    
    return score_objectifs

def audit_strategique_list(user_answers, question_ids):
    '''
    Évaluer les réponses de l'utilisateur pour une liste de questions et retourner les scores totaux pour chaque objectif.

    Input:
        user_answers (list): Liste des réponses de l'utilisateur.
        question_ids (list): Liste des IDs des questions correspondantes.

    Output:
        dict: Un dictionnaire avec les objectifs comme clés et les scores totaux correspondants comme valeurs.
    '''
    objectifs = objectif_df['Objectif'].unique()
    score_objectifs = {objectif: 0 for objectif in objectifs}
    
    if len(user_answers) != len(question_ids):
        raise ValueError("La longueur de user_answers doit être égale à celle de question_ids")
    
    for user_answer, question_id in zip(user_answers, question_ids):
        answer_scores = audit_strategique(user_answer, question_id)
        for objectif in objectifs:
            score_objectifs[objectif] += answer_scores[objectif]
    
    return score_objectifs

# Charger les données des axes et des scores
axes_df = pd.read_csv("./audit_digital_axes.csv")
answers_df = pd.read_csv("./audit_digital_scores.csv")

def auditdigitale(user_answers, question_ids):
    '''
    Évaluer les réponses de l'utilisateur pour une liste de questions et retourner les scores totaux pour chaque axe.

    Input:
        user_answers (list): Liste des réponses de l'utilisateur.
        question_ids (list): Liste des IDs des questions correspondantes.

    Output:
        dict: Un dictionnaire avec les axes comme clés et les scores totaux correspondants comme valeurs.
    '''
    axes = axes_df['Axis'].unique()
    score_axes = {axis: 0 for axis in axes}
    
    if len(user_answers) != len(question_ids):
        raise ValueError("La longueur de user_answers doit être égale à celle de question_ids")
    
    for user_answer, question_id in zip(user_answers, question_ids):
        user_score = calculate_score(user_answer, question_id, answers_df=answers_df, questions_df=axes_df)
        question_axis = axes_df[axes_df['ID'] == question_id]['Axis'].values
        if len(question_axis) == 0:
            raise ValueError(f"Aucun axe trouvé pour l'ID de question : {question_id}")
        
        axis = question_axis[0]
        score_axes[axis] += user_score
    
    return score_axes

def create_radar_chart(score_axes, filename='static/scores_plot.png'):
    '''
    Générer un graphique radar pour les scores des axes et le sauvegarder en tant que fichier image.

    Input:
        score_axes (dict): Un dictionnaire avec les axes comme clés et les scores correspondants comme valeurs.
        filename (str): Nom du fichier image où le graphique sera sauvegardé.
    '''
    # Convertir le dictionnaire en DataFrame pour faciliter la création du graphique
    df_scores = pd.DataFrame(list(score_axes.items()), columns=['Axis', 'Score'])
    
    # Radar chart parameters
    labels = df_scores['Axis'].tolist()
    values = df_scores['Score'].tolist()
    num_vars = len(labels)

    # Radar chart angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    plt.xticks(angles[:-1], labels, color='grey', size=8)
    plt.title('Scores par Axe', size=20, color='black', y=1.1)

    # Sauvegarder le graphique en tant que fichier image
    plt.savefig(filename, format='png')
    plt.close()  # Fermer la figure pour libérer de la mémoire

    print(f"Graphique sauvegardé sous le nom de fichier : {filename}")

if __name__ == "__main__":
    # Exemple d'utilisation
    user_answers = [
        "Marketing",
        "non"
    ]
    question_ids = [
        1,  # Remplacez avec l'ID réel de la question
        2   # Remplacez avec l'ID réel de la question
    ]

    scores = audit_strategique_list(user_answers, question_ids)
    for objectif, score in scores.items():
        print(f"Score total pour l'objectif '{objectif}': {score:.2f}")

    # Exemple d'utilisation
    user_answers = [
        "Marketing",
        "non"
    ]
    question_ids = [
        1,  # Remplacez avec l'ID réel de la question
        2   # Remplacez avec l'ID réel de la question
    ]

    scores = auditdigitale(user_answers, question_ids)
    create_radar_chart(scores)
    for axis, score in scores.items():
        print(f"Score total pour l'axe '{axis}': {score:.2f}")
