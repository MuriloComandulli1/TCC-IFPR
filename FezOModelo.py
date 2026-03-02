import numpy as np
import pandas as pd
from scipy.io import arff

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Carregar arquivo ARFF
def load_arff(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    # Converter atributos categóricos de bytes para strings
    df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    return df

# Função principal para treinar e salvar o modelo
def train_and_save_model(data_path, output_dir):
    # Carregar os dados
    df = load_arff(data_path)

    # Separar features e labels
    X = df.drop(columns=["label"])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])  # Codifica as labels automaticamente

    # Definir hiperparâmetros para Grid Search
    knn_params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    
    # Split correto
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # GridSearch KNN
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_params,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
    )
    knn_grid.fit(X_train, y_train)

    print("Melhor KNN:", knn_grid.best_params_)
    print("Melhor F1 (CV):", knn_grid.best_score_)

    # GridSearch RF
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)

    print("Melhor RF:", rf_grid.best_params_)
    print("Melhor F1 (CV):", rf_grid.best_score_)

    # Definir o modelo Stacking
    stacking_model = StackingClassifier(
        estimators=[('knn', knn_grid.best_estimator_), ('rf', rf_grid.best_estimator_)], 
        final_estimator=LogisticRegression()
    )


    # Stacking
    stacking_model.fit(X_train, y_train)

    # Avaliação final
    y_pred = stacking_model.predict(X_test)

    print("\n===== RESULTADO FINAL =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred, average='weighted'))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Criar uma pasta para salvar o modelo
    os.makedirs(output_dir, exist_ok=True)

    # Salvar o modelo e o LabelEncoder
    model_path = os.path.join(output_dir, "stacking_model.pkl")
    label_encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    joblib.dump(stacking_model, model_path)
    joblib.dump(label_encoder, label_encoder_path)

    print(f"Modelo salvo em: {model_path}")
    print(f"LabelEncoder salvo em: {label_encoder_path}")

# Exemplo de execução
if __name__ == "__main__":
    # Caminho para os dados e pasta de saída
    data_path = ""
    output_dir = "./modeloteste"

    # Treinar e salvar o modelo
    train_and_save_model(data_path, output_dir)
    print(f"modelo salvo com sucesso em {data_path}")