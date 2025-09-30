import streamlit as st 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import mode
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Charger dataset IRIS
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Créer et entraîner les modèles
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_model.fit(X)

# --- Application ---
st.set_page_config(page_title="Classification IRIS", layout="wide")

# Menu latéral
st.sidebar.markdown("<h1 style='text-align: center; color: #FFFFFFFF;'>Navigation</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")  # Séparateur

# Menu
Navigation = ["Accueil", "Classification", "Évaluation des modèles", "Corrélation"]
choice = st.sidebar.selectbox("", Navigation)
# --- Accueil ---
if choice == "Accueil":
    st.title("IRISApp - Classification et Analyse des Fleurs")
    image = Image.open("iris.jpg")  # Remplace par ton chemin
    st.image(image, caption="Fleurs IRIS", use_container_width=True)
    st.markdown("""
<div style="background-color: #B0B2F0FF; padding: 20px; border-radius: 10px;">
    <h3 style="color: #031F3CFF;">Cette application permet de :</h3>
    <ul style="list-style-type: square; padding-left: 20px; font-size: 16px; color: #000000FF;">
        <li>Classifier une fleur IRIS selon ses 4 caractéristiques.</li>
        <li> Évaluer les différents modèles (Régression Logistique, SVM, KNN et KMeans).</li>
        <li> Visualiser la corrélation entre les caractéristiques.</li>
        <li> Observer le clustering KMeans intégré à la prédiction.</li>
    </ul>
</div>
""", unsafe_allow_html=True)
# --- Prédiction ---
elif choice == "Classification":
    st.markdown("""
    <div style="
        background-color: #191819FF; 
        padding: 30px; 
        border-radius: 15px; 
        box-shadow: 2px 2px 15px rgba(0,0,0,0.3);
    ">
        <h2 style="color: #FFFFFF;">Classification  d'une nouvelle fleur IRIS</h2>
        <p style="color: #A06ED2FF; font-size:16px;">Saisissez les caractéristiques de la fleur (0 à 10 cm) :</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Card Inputs ---
 

    # Inputs en colonnes à l'intérieur de la card
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Longueur du sépale (cm)", 0.0, 10.0, 5.1, 0.1, key='sl')
        sepal_width  = st.number_input("Largeur du sépale (cm)", 0.0, 10.0, 3.5, 0.1, key='sw')
    with col2:
        petal_length = st.number_input("Longueur du pétale (cm)", 0.0, 10.0, 1.4, 0.1, key='pl')
        petal_width  = st.number_input("Largeur du pétale (cm)", 0.0, 10.0, 0.2, 0.1, key='pw')

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Choix du modèle
    model_choice = st.selectbox(
        "Choisissez le modèle :", 
        ["Régression Logistique", "SVM", "KNN", "KMeans"]
    )

    # Bouton centralisé
    if st.button("Lancer la classification"):
        if model_choice == "Régression Logistique":
            prediction = log_model.predict(features)
            st.success(f"Classe prédite : **{target_names[prediction][0]}**")
        elif model_choice == "SVM":
            prediction = svm_model.predict(features)
            st.success(f"Classe prédite : **{target_names[prediction][0]}**")
        elif model_choice == "KNN":
            prediction = knn_model.predict(features)
            st.success(f"Classe prédite : **{target_names[prediction][0]}**")
        elif model_choice == "KMeans":
            cluster = kmeans_model.predict(features)
            st.info(f"Cluster attribué : **{cluster[0]}**")
            st.warning("⚠️ KMeans est non supervisé : clusters ≠ classes réelles.")

            # --- Card KMeans plot ---
            st.markdown("""
            <div style="
                background-color: #000000FF; 
                padding: 20px; 
                border-radius: 10px; 
            ">
                <h3 style="color:#FFFFFF;">Visualisation des clusters KMeans</h3>
            </div>
            """, unsafe_allow_html=True)

            clusters = kmeans_model.predict(X)
            fig, ax = plt.subplots()
            colors = ['red', 'green', 'blue']
            for i in range(3):
                ax.scatter(
                    X.iloc[clusters==i, 0],
                    X.iloc[clusters==i, 2],
                    label=f'Cluster {i}',
                    c=colors[i],
                    alpha=0.6
                )
            ax.scatter(
                features[0, 0],
                features[0, 2],
                color='black',
                edgecolor='yellow',
                s=200,
                label='Votre fleur'
            )
            ax.set_xlabel("Sepal length")
            ax.set_ylabel("Petal length")
            ax.legend()
            st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)  # fermer la card

# --- Évaluation des modèles ---
elif choice == "Évaluation des modèles":
    st.header("📊 Évaluation des modèles supervisés et KMeans")

    from scipy.stats import mode

    # Fonction pour transformer les clusters en labels majoritaires
    def kmeans_accuracy_labels(y_true, y_pred):
        labels = np.zeros_like(y_pred)
        for i in range(3):  # 3 clusters
            mask = (y_pred == i)
            if np.sum(mask) == 0:
                continue
            labels[mask] = mode(y_true[mask])[0]
        return labels

    # Fonction principale d'évaluation
    def show_evaluation(model, X_test, y_test, name):
        y_pred = model.predict(X_test)
        st.subheader(f"Modèle : {name}")

        # Pour KMeans : transformer les clusters en labels majoritaires
        if isinstance(model, KMeans):
            y_pred_labels = kmeans_accuracy_labels(y_test, y_pred)
        else:
            y_pred_labels = y_pred

        # Accuracy
        acc = accuracy_score(y_test, y_pred_labels)
        st.write(f"**Accuracy:** {acc:.2f}")
        col1, col2 = st.columns([2, 1])  # Table gets more space
        # Classification report comme DataFrame
        report_dict = classification_report(
            y_test, y_pred_labels, target_names=target_names, output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        st.write("**Classification Report (Tableau) :**")
        st.dataframe(report_df.style.set_table_styles(
            [{'selector': 'th', 'props': [('font-size', '16px')]},
             {'selector': 'td', 'props': [('font-size', '14px')]}]
        ).format("{:.2f}"))

        # Matrice de confusion compacte
        st.write("**Matrice de confusion :**")
        cm = confusion_matrix(y_test, y_pred_labels)
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=target_names, yticklabels=target_names, ax=ax
        )
        ax.set_xlabel("Prédite")
        ax.set_ylabel("Réel")
        st.pyplot(fig)

    # Évaluer tous les modèles
    show_evaluation(log_model, X_test, y_test, "Régression Logistique")
    show_evaluation(svm_model, X_test, y_test, "SVM")
    show_evaluation(knn_model, X_test, y_test, "KNN")
    show_evaluation(kmeans_model, X_test, y_test, "KMeans")



# --- Corrélation ---
elif choice == "Corrélation":
    st.header("📈 Matrice de corrélation des caractéristiques")
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(
        X.corr(), 
        annot=True, 
        cmap="coolwarm", 
        ax=ax, 
        cbar=False,       # remove colorbar
        annot_kws={"size": 8}  # smaller numbers
    )
    plt.xticks(fontsize=8, rotation=45)
    plt.yticks(fontsize=8, rotation=0)
    plt.tight_layout()  # ensure it fits nicely
    st.pyplot(fig)