import pandas as pd
import seaborn as sns


df=pd.read_csv('/content/Parkinsons.csv')
df.head()

df.shape

df.isnull().sum()

df.columns.tolist()

df.dtypes

df.info()

df.describe().round(2)

import matplotlib.pyplot as plt
corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(15,12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Matrice de corrélation")
plt.show()

df.duplicated().sum()

features = ['age', 'sex', 'test_time', 'jitter', 'jitter_abs', 'shimmer', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']
X = df[features].copy()
y_motor = df['motor_updrs']
y_total = df['total_updrs']

X['sex'] = X['sex'].astype(int)


for feature in features:
    if feature != 'sex':
        print(f"{feature:12}: min={X[feature].min():.4f}, max={X[feature].max():.4f}")

features_a_normaliser=[f for f in features if f != 'sex']
features_a_normaliser


from sklearn.preprocessing import StandardScaler,LabelEncoder
import joblib

scaler = StandardScaler()
X_normalise = X.copy()
X_normalise[features_a_normaliser] = scaler.fit_transform(X[features_a_normaliser])
print("f",X_normalise[features_a_normaliser])
print("les x",X)
joblib.dump(scaler,'scaler.pkl')




def categoriser_gravite(updrs):
    if updrs < 30:
        return "Stable"
    elif updrs < 45:
        return "Modéré"
    else:
        return "Grave"

y_gravite = df['total_updrs'].apply(categoriser_gravite)
y_gravite

y_gravite.value_counts()

y_gravite.value_counts(normalize=True).round(2)

from sklearn.model_selection import train_test_split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_normalise, y_total, test_size=0.3, random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_normalise, y_gravite, test_size=0.3,stratify=y_gravite ,random_state=42
)


from sklearn.neighbors import KNeighborsRegressor
knn_updrs = KNeighborsRegressor(n_neighbors=5)
knn_updrs.fit(X_train_reg, y_train_reg)
y_pred_updrs = knn_updrs.predict(X_test_reg)

comparaison = pd.DataFrame({
    'Réel': y_test_reg.values[:8],
    'Prédit': y_pred_updrs[:8].round(1)
})
comparaison['Différence'] = (comparaison['Réel'] - comparaison['Prédit']).abs()
comparaison

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

knn_gravite = KNeighborsClassifier(n_neighbors=5)
knn_gravite.fit(X_train_clf, y_train_clf)
y_pred_gravite = knn_gravite.predict(X_test_clf)

accuracy=accuracy_score(y_test_clf,y_pred_gravite)
print("accuracy= ",accuracy)

confusion_matrix=confusion_matrix(y_test_clf,y_pred_gravite)
print("la matrice de confusion",confusion_matrix)

print("details",classification_report(y_test_clf,y_pred_gravite))

scores_k=[]
k_values= range(1,21)
for k in k_values:
  model=KNeighborsClassifier(n_neighbors=k)
  model.fit(X_train_clf,y_train_clf)
  print("k=",k,"score=",model.score(X_test_clf,y_test_clf))
  scores_k.append(model.score(X_test_clf,y_test_clf))
meilleure_k=k_values[scores_k.index(max(scores_k))]
meilleur_score=max(scores_k)
print(f"pour notre modele la meilleure k est {meilleure_k} avec une score {meilleur_score} ")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(k_values, scores_k, marker='o', linewidth=2, markersize=6, color='blue')
plt.title("Optimisation du paramètre K - Classification Gravité")
plt.xlabel("Valeur de K")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.axvline(x=meilleure_k, color='red', linestyle='--', label=f'Meilleur K={meilleure_k}')
plt.legend()
plt.savefig('optimisation_k.png', dpi=300, bbox_inches='tight')
plt.show()

#NV patient
nouveau_patient = [72, 1, 50, 0.005, 0.00003, 0.025, 0.015, 25, 0.5, 0.55, 0.16]

nouveau_df = pd.DataFrame([nouveau_patient], columns=features)
nouveau_df[features_a_normaliser] = scaler.transform(nouveau_df[features_a_normaliser])

# Préd
updrs_pred = knn_updrs.predict(nouveau_df)[0]
gravite_pred = knn_gravite.predict(nouveau_df)[0]

print(f"Resultat de  prediction :")
print(f"UPDRS total prédit : {updrs_pred:.1f}")
print(f"Niveau de gravité : {gravite_pred}")

#systeme d'alerte
def evaluer_urgence(updrs,gravite):
  if gravite=="Stable":
    return "Stable-controle dans 3 mois"
  elif gravite=="Modéré":
    return "Modere-controle dans 1 mois"
  else:
    return "urgence-contacter le medecin immediatement"
alerte=evaluer_urgence(updrs_pred,gravite_pred)




import os

from google.colab import files

joblib.dump(knn_updrs, 'model_reg.pkl')
joblib.dump(knn_gravite, 'model_clf.pkl')
joblib.dump(scaler, 'scaler.pkl')

df.to_csv('Parkinsons.csv', index=False)

with open('requirements.txt', 'w') as f:
    f.write('''pandas==1.5.3
scikit-learn==1.2.2
joblib==1.2.0
matplotlib==3.7.1
seaborn==0.12.2
streamlit==1.28.0
''')

!zip -r parkinson_project.zip ./*.pkl ./Parkinsons.csv ./requirements.txt

files.download('parkinson_project.zip')
