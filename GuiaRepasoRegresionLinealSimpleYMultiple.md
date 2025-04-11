
# 🧠 Guía de Repaso: Regresión Lineal en Google Colab

## 🎯 Objetivo
Practicar regresión lineal simple, múltiple, evaluación del modelo y regularización (Ridge y Lasso) usando Python y Google Colab.

---

## 🚀 Instrucciones
1. Abre [Google Colab](https://colab.research.google.com/)
2. Crea un nuevo cuaderno.
3. Copia y pega los siguientes bloques uno por uno y sigue las instrucciones.

---

## 📊 1. Cargar los datos

```python
import pandas as pd

datos = {
    "tamaño_m2": [50, 60, 80, 100, 120, 150, 200],
    "num_habitaciones": [1, 2, 2, 3, 3, 4, 5],
    "precio": [800, 1000, 1300, 1600, 1900, 2300, 3000]
}

df = pd.DataFrame(datos)
df
```

---

## 📈 2. Gráfico de dispersión

```python
import matplotlib.pyplot as plt

plt.scatter(df["tamaño_m2"], df["precio"])
plt.xlabel("Tamaño (m2)")
plt.ylabel("Precio (mil pesos)")
plt.title("Relación entre tamaño y precio")
plt.show()
```

---

## 📐 3. Regresión Lineal Simple

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X_simple = df[["tamaño_m2"]]
y = df["precio"]

modelo_simple = LinearRegression()
modelo_simple.fit(X_simple, y)

print("Ecuación: Precio = {:.2f} * Tamaño + {:.2f}".format(
    modelo_simple.coef_[0], modelo_simple.intercept_))

prediccion = modelo_simple.predict([[110]])
print("Predicción para 110 m²:", prediccion[0])
```

---

## 🧮 4. Regresión Lineal Múltiple

```python
X_multiple = df[["tamaño_m2", "num_habitaciones"]]

modelo_multiple = LinearRegression()
modelo_multiple.fit(X_multiple, y)

from sklearn.metrics import mean_squared_error, r2_score

y_pred = modelo_multiple.predict(X_multiple)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("MSE:", mse)
print("R²:", r2)
```

---

## 🧪 5. Regularización (Ridge y Lasso)

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)
ridge.fit(X_multiple, y)
print("Coeficientes Ridge:", ridge.coef_)

lasso = Lasso(alpha=1.0)
lasso.fit(X_multiple, y)
print("Coeficientes Lasso:", lasso.coef_)
```

---

## 💭 6. Bonus: Predicciones

```python
nuevas_casas = pd.DataFrame({
    "tamaño_m2": [90, 150, 210],
    "num_habitaciones": [2, 3, 5]
})

predicciones = modelo_multiple.predict(nuevas_casas)
print("Predicciones para nuevas casas:\n", predicciones)
```

---

## ✅ Fin de la Guía
Esta guía es solo para práctica personal. No es necesario entregarla.
