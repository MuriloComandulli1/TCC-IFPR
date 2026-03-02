from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from scipy.io import arff
import numpy as np

data, meta = arff.loadarff('features_earlyfusion.arff')
df = pd.DataFrame(data)
for column in df.select_dtypes([np.object_]).columns:
    df[column] = df[column].str.decode('utf-8')

X = df.drop('label', axis=1) 
y = df['label']

# Codificar os rótulos de string para valores binários
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Verificar os rótulos após a codificação
print("Rótulos após codificação:", label_encoder.classes_)

# Definir o modelo
knn = KNeighborsClassifier()

# Definir o grid de parâmetros
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13],
    'weights': ['uniform', 'distance'],
}

# Certifique-se de que o f1_score está sendo configurado corretamente para problemas binários
scorer = make_scorer(f1_score, average='macro', pos_label=1)

# Configurar o GridSearchCV
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, scoring=scorer, cv=10)

# Executar a pesquisa
grid_search.fit(X, y_encoded)

# Exibir os melhores parâmetros encontrados
print("Melhores parâmetros:", grid_search.best_params_)

# Exibir a melhor pontuação obtida
print("Melhor F1-Score:", grid_search.best_score_)
