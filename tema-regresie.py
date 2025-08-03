import numpy as np
from PIL.GimpGradientFile import linear
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import joblib

x = np.array([

[1, 20, 1.2],

[2, 35, 1.4],

[3, 60, 1.6],

[4, 80, 1.6],

[5, 120, 1.8],

[6, 150, 2.0],

[7, 180, 2.0],

])
y = np.array([12000, 11000, 9500, 8500, 7000, 6000, 5000])

model = LinearRegression()

model.fit(x,y)

#r2 apartine [0,1]- cu cat r2 este mai aproape de 1 cua atat predictia este mai buna
r2 = model.score(x,y)
print(r2)


print("Coeficienti (beta1, beta2): ", model.coef_)
print("Interceptul beta0: ", model.intercept_)

pret = model.predict([[4,75, 1.6]])
print(f"Predictie pentru 4, 75.000 km , 1.6l :{pret}")


#Predictie date user - Documentatie - https://stackoverflow.com/questions/62938642/how-to-take-input-as-an-array-from-user-in-python-using-loops
string = input("Te rog insereaza datele masinii tale: ")
arr = list(map(float, string.split(' ')))
pret_user = model.predict([arr])
print(f"Predictia pentru inputul userului este: {pret_user}")

# dump(value, filename[, compress, protocol])- Documentatie https://joblib.readthedocs.io/en/stable/
joblib.dump(model, 'regresie_pret_masini.pkl')
print("Modelul a fost salvat sub forma de fisier .pkl")

vechime_range = np.linspace(x[:, 0].min(), x[:, 0].max(), 30)
rulaj_range = np.linspace(x[:, 1].min(), x[:, 1].max(), 30)
vechime_grid, rulaj_grid = np.meshgrid(vechime_range, rulaj_range)
capacitate_fix = np.mean(x[:, 2])

# Pregătim datele pentru predicție pe grilă
X_grid = np.column_stack((vechime_grid.ravel(), rulaj_grid.ravel(), np.full(vechime_grid.size, capacitate_fix)))
pret_pred = model.predict(X_grid).reshape(vechime_grid.shape)

# Creare grafic 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Puncte reale colorate distinct în funcție de preț
scatter = ax.scatter(x[:, 0], x[:, 1], y, c=y, cmap='viridis', s=50, label='Date reale')

# Planul de regresie
ax.plot_surface(vechime_grid, rulaj_grid, pret_pred, color='skyblue', alpha=0.5, label='Plan regresie')

# Etichete axe
ax.set_xlabel('Vechime (ani)')
ax.set_ylabel('Rulaj (mii km)')
ax.set_zlabel('Preț (EUR)')
ax.set_title('Regresie Liniară 3D: Vechime vs Rulaj vs Preț')

# Bară de culori pentru preț
fig.colorbar(scatter, ax=ax, label='Preț')

plt.show()