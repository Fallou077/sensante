import joblib
import pandas as pd

model = joblib.load("models/model.pkl")
feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys', 'toux', 'fatigue', 'maux_tete', 'region_encoded']

# 2. Création des 3 patients fictifs
# Rappel de l'ordre : [age, sexe(0=F,1=M), temp, tension, toux(0/1), fatigue(0/1), maux_tete(0/1), region(0...)]
data_patients = [
    [22, 1, 36.6, 12, 0, 0, 0, 1], # Jeune, sans symptômes (Sain ?)
    [45, 1, 39.8, 14, 1, 1, 1, 1], # Adulte, forte fièvre (Paludisme/Grippe ?)
    [78, 0, 38.2, 11, 1, 0, 0, 1]  # Patient âgé avec toux
]

# 3. Transformer en DataFrame pour éviter les warnings
df_test = pd.DataFrame(data_patients, columns=feature_cols)

# 4. Prédire
predictions = model.predict(df_test)
probabilites = model.predict_proba(df_test)

# 5. Afficher les résultats
print("\n--- Résultats des Tests (Exo 2) ---")
types_patients = ["Jeune sain", "Adulte fiévreux", "Âgé avec toux"]

for i, pred in enumerate(predictions):
    confiance = max(probabilites[i]) * 100
    print(f"Patient {i+1} ({types_patients[i]}) :")
    print(f"   > Diagnostic : {pred}")
    print(f"   > Confiance : {confiance:.1f}%\n")