import matplotlib.pyplot as plt
import numpy as np

# Data for each axis
data = {
    "Stratégie Digitale": [
        {"question": "Votre entreprise a-t-elle une stratégie digitale clairement définie ?", "scores": [10, 0, 5]},
        {"question": "Avez-vous fixé des objectifs mesurables pour votre stratégie digitale ?", "scores": [10, 0, 5]},
        {"question": "Est-ce que votre entreprise suit les tendances technologiques et numériques ?", "scores": [10, 0, 5]},
        {"question": "Avez-vous alloué un budget spécifique pour les initiatives digitales ?", "scores": [10, 0, 5]},
    ],
    "Marketing Digital": [
        {"question": "Votre entreprise utilise-t-elle des outils de marketing automation ?", "scores": [10, 0, 5]},
        {"question": "Avez-vous une stratégie de contenu pour votre blog ou site web ?", "scores": [10, 0, 5]},
        {"question": "Est-ce que vous mesurez régulièrement l'efficacité de vos campagnes marketing digitales ?", "scores": [10, 0, 5]},
        {"question": "Votre entreprise utilise-t-elle des techniques de ciblage publicitaire avancées, comme le retargeting ?", "scores": [10, 0, 5]},
    ],
    "Transformation Digitale": [
        {"question": "Avez-vous intégré des outils numériques pour améliorer vos processus internes ?", "scores": [10, 0, 5]},
        {"question": "Est-ce que vos employés sont formés aux nouvelles technologies et outils digitaux ?", "scores": [10, 0, 5]},
        {"question": "Utilisez-vous des solutions cloud pour vos opérations ?", "scores": [10, 0, 5]},
        {"question": "Votre entreprise a-t-elle un plan de transformation digitale en cours ?", "scores": [10, 0, 5]},
    ],
    "Expérience Client Digitale": [
        {"question": "Votre site web offre-t-il une expérience utilisateur optimisée ?", "scores": [10, 0, 5]},
        {"question": "Avez-vous mis en place des outils de gestion de la relation client (CRM) ?", "scores": [10, 0, 5]},
        {"question": "Utilisez-vous des chatbots ou assistants virtuels pour améliorer le support client ?", "scores": [10, 0, 5]},
        {"question": "Est-ce que vous recueillez et analysez régulièrement les feedbacks des clients pour améliorer vos services ?", "scores": [10, 0, 5]},
    ],
}

# Function to create a radar chart
def create_radar_chart(title, data):
    labels = [d["question"] for d in data]
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], labels, color='grey', size=8)
    ax.set_rlabel_position(0)
    plt.yticks([0, 5, 10], ["0", "5", "10"], color="grey", size=7)
    plt.ylim(0, 10)

    for scores in zip(*[d["scores"] for d in data]):
        values = list(scores)
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)

    plt.title(title, size=20, color='black', y=1.1)
    plt.show()

# Create radar charts for each axis
for axis, questions in data.items():
    create_radar_chart(axis, questions)
