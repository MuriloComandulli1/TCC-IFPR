from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from scipy.io import arff
import numpy as np

# Carregar os dados
data, meta = arff.loadarff('features_earlyfusion.arff')
df = pd.DataFrame(data)
for column in df.select_dtypes([np.object_]).columns:
    df[column] = df[column].str.decode('utf-8')

# Separando as variáveis independentes (X) e dependentes (y)
X = df.drop('label', axis=1)
y = df['label']

# Codificar os rótulos de string para valores binários
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Verificar os rótulos após a codificação
print("Rótulos após codificação:", label_encoder.classes_)

# Definir o modelo SVM
svm = SVC()

# Definir o grid de parâmetros
param_grid = {
    'C': [0.1, 1, 10],
}

# Certifique-se de que o f1_score está sendo configurado corretamente para problemas binários
scorer = make_scorer(f1_score, average='macro', pos_label=1)

# Configurar o GridSearchCV
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, scoring=scorer, cv=10)

# Executar a pesquisa
grid_search.fit(X, y_encoded)

print("Melhores parâmetros:", grid_search.best_params_)
print("Melhor F1-Score:", grid_search.best_score_)