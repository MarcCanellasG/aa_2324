import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from Adaline import Adaline
from sklearn.svm import SVC 


# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

# Separar les dades: train_test_split
# TODO

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

# Estandaritzar les dades: StandardScaler
# TODO

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

scaler = StandardScaler()
x_test = scaler.fit_transform(x_test)

# Entrenam una SVM linear (classe SVC)
# TODO
modelo= SVC (C=10000 , kernel='linear')
modelo.fit(x_train,y_train)

# Prediccio
# TODO
result = modelo.predict(x_test)

# Metrica
# TODO
acerts=np.sum(np.abs(result-y_test))
mostres=len(result)
print(mostres)
print((mostres-acerts) / mostres)