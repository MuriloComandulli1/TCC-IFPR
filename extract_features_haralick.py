import os
import numpy as np
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import time

# Função para extrair características de Haralick
def extract_haralick_features(image):
    # Calcular a matriz de coocorrência de Haralick
    glcm = graycomatrix(image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    
    # Extrair características de Haralick
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    ASM = graycoprops(glcm, 'ASM')

    # Retornar uma lista com todas as características extraídas
    return [contrast.mean(), dissimilarity.mean(), homogeneity.mean(), energy.mean(), correlation.mean(), ASM.mean()]

# Função para processar imagens de uma pasta
def process_images_from_folder(folder, label):
    features = []
    labels = []
    image_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Exibe o progresso do processamento das imagens
    for filename in tqdm(image_files, desc=f"Processando {label} em {folder}"):
        img_path = os.path.join(folder, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Lê a imagem em escala de cinza
        
        # Verifica se a imagem é 2D
        if len(image.shape) == 2:
            # Extrai as características de Haralick
            haralick_features = extract_haralick_features(image)
            features.append(haralick_features)
            labels.append(label)
        else:
            print(f"A imagem {filename} não está em escala de cinza.")
    
    return features, labels

# Função para unir dados de múltiplas pastas e gerar CSV
def process_all_folders():
    folders = [
      ('./Dataset_augmented_V2/CT_Tumor', 'tumor'),  # pasta com imagens de tomografia (tumor)
        ('./Dataset_augmented_V2/CT_Healthy', 'healthy'),  # pasta com imagens de tomografia (saudáveis)
        ('./Dataset_augmented_V2/MRI_Tumor', 'tumor'),  # pasta com imagens de MRI (tumor)
        ('./Dataset_augmented_V2/MRI_Healthy', 'healthy')  # pasta com imagens de MRI (saudáveis)
    ]
    
    all_features = []
    all_labels = []
    
    # Processa todas as pastas
    for folder, label in folders:
        folder_features, folder_labels = process_images_from_folder(folder, label)
        all_features.extend(folder_features)
        all_labels.extend(folder_labels)
    
    # Cria o DataFrame com as características e os rótulos
    df = pd.DataFrame(all_features, columns=["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"])
    df["label"] = all_labels
    
    # Codifica os rótulos (saudável/tumor) para valores numéricos
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])
    print("Mapeamento dos rótulos:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    
    # Salva o DataFrame como CSV
    df.to_csv("features_haralick.csv", index=False)

# Chama a função para processar as imagens e gerar o CSV
process_all_folders()

print("Processamento concluído e CSV gerado com as características de Haralick.")
