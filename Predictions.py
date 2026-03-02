import joblib
import numpy as np
from ModelLogic.image_processing import process_single_image
import os



def predict_image_with_model(image_path):
    """
    Processa uma imagem, extrai suas características, usa um modelo pré-treinado
    para prever o rótulo e exibe a porcentagem de confiança da previsão.

    Args:
        image_path (str): Caminho para a imagem a ser testada.
        model_path (str): Caminho para o modelo treinado salvo.
        label_encoder_path (str): Caminho para o LabelEncoder salvo.

    Returns:
        None
    """
    model_path = "backend/model/stacking_model.pkl"
    label_encoder_path = "backend/model/label_encoder.pkl"
    # Carregar o modelo treinado e o LabelEncoder
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    # Extrair características da imagem
    try:
        features = process_single_image(image_path)
    except ValueError as e:
        print("Erro ao processar a imagem:", e)
        return

    # Converter as características em um vetor de entrada para o modelo
    feature_vector = np.array(list(features.values())).reshape(1, -1)

    # Fazer a previsão
    predictions = model.predict(feature_vector)
    probabilities = model.predict_proba(feature_vector)

    # Decodificar o rótulo previsto
    predicted_label = label_encoder.inverse_transform(predictions)[0]
    predicted_probability = probabilities[0][predictions[0]] * 100

    # Exibir os resultados
    print(f"Rótulo previsto: {predicted_label}")
    print(f"Confiança na previsão: {predicted_probability:.2f}%")
    return {
            "predicted_label": predicted_label,
            "predicted_probability": round(predicted_probability, 2)
        }

# Exemplo de uso
if __name__ == "__main__":
    # Caminhos necessários
    # Caminho baseado na localização do script
    

    #TESTE
    ####
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "test/testesaudavel.jpeg")
    print("Caminho absoluto gerado:", image_path)
    ####


 
    # Fazer a previsão
    predict_image_with_model(image_path)
