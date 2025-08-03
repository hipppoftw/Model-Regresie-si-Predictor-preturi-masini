import joblib
import numpy as np

# https://joblib.readthedocs.io/en/latest/generated/joblib.load.html#joblib.load
model = joblib.load("regresie_pret_masini.pkl")
print("Modelul a fost incarcat")


#Copiat logica anterioara
string = input("Te rog insereaza datele masinii tale: ")
arr = list(map(float, string.split(' ')))
pret_user = model.predict([arr])
print(f"Predictia pentru inputul userului este: {pret_user}")

