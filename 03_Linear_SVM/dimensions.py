import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Adaline import Adaline
from sklearn.svm import SVC 


# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

# Separar les dades: train_test_split
# TODO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Estandaritzar les dades: StandardScaler
# TODO
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

scaler_test = StandardScaler()
X_test = scaler_test.fit_transform(X_test)

# Entrenam una SVM linear (classe SVC)
# TODO
model = SVC(C = 10000 , kernel = 'linear')
model.fit(X_train,y_train)
# Prediccio
# TODO
result = model.predict(X_test)

# Metrica
# TODO
acerts=np.sum(np.abs(result-y_test))
mostres=len(result)
print(mostres)
print((mostres-acerts) / mostres)