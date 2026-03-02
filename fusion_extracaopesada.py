import os
import numpy as np
import cv2
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from tqdm import tqdm
import random

def extract_haralick_features(image):
    glcm = graycomatrix(image, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    features = [graycoprops(glcm, prop).mean() for prop in ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM')]
    return features

def extract_lbp_features(image, P, R):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return lbp_hist.tolist()

def extract_features(image, P, R):
    haralick = extract_haralick_features(image)
    lbp = extract_lbp_features(image, P, R)
    return haralick + lbp

def process_images_from_folder(folder, label, P, R):
    features = []
    labels = []
    image_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(image_files, desc=f"Processando {label} (P={P}, R={R})"):
        img_path = os.path.join(folder, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is not None and len(image.shape) == 2:
            combined_features = extract_features(image, P, R)
            features.append(combined_features)
            labels.append(label)
        else:
            print(f"Erro ao processar: {filename}")
    
    return features, labels

def save_as_arff(csv_filename, arff_filename):
    df = pd.read_csv(csv_filename)
    with open(arff_filename, 'w') as f:
        f.write("@RELATION features\n\n")
        for col in df.columns[:-1]:
            f.write(f"@ATTRIBUTE {col} NUMERIC\n")
        f.write("@ATTRIBUTE label {tumor,healthy}\n\n")
        f.write("@DATA\n")
        df.to_csv(f, index=False, header=False)

def process_all_folders():
    folders = [
        ('Dataset_full_Augmented/Brain Tumor_CT_scan_Images/Healthy', 'healthy'),
        ('Dataset_full_Augmented/Brain Tumor_CT_scan_Images/Tumor', 'tumor'),
        ('Dataset_full_Augmented/Brain_Tumor_MRI_images/Healthy', 'healthy'),
        ('Dataset_full_Augmented/Brain_Tumor_MRI_images/Tumor', 'tumor'),
        ('Dataset_full_Augmented/glioma', 'tumor'),
        ('Dataset_full_Augmented/meningioma', 'tumor'),
        ('Dataset_full_Augmented/pituitary', 'tumor'),
        ('Dataset_full_Augmented/notumor', 'healthy'),
    ]
    
    output_dir = r"C:\Users\User\Desktop\Murilo_Ricardo_Pedro\LBPHaralickTests"
    os.makedirs(output_dir, exist_ok=True)
    
    P_values = [2, 4, 6, 8, 10, 12]
    R_values = [3, 5, 7, 9, 11, 13
                ]
    
    combinations = [(P, R) for P in P_values for R in R_values]
    random.shuffle(combinations)
    combinations = combinations[:20]  # Limita a 10 combinações se houver muitas
    
    processed_combinations = set()
    
    for P, R in combinations:
        if (P, R) in processed_combinations:
            continue
        processed_combinations.add((P, R))
        
        all_features = []
        all_labels = []
        
        for folder, label in folders:
            folder_features, folder_labels = process_images_from_folder(folder, label, P, R)
            all_features.extend(folder_features)
            all_labels.extend(folder_labels)
        
        if all_features:
            feature_names = [f"haralick_{i}" for i in range(6)] + [f"lbp_{i}" for i in range(len(all_features[0]) - 6)]
            df = pd.DataFrame(all_features, columns=feature_names)
            df["label"] = all_labels
            
            csv_filename = os.path.join(output_dir, f"features_EarlyFusion_P{P}_R{R}.csv")
            arff_filename = os.path.join(output_dir, f"features_EarlyFusion_P{P}_R{R}.arff")
            
            df.to_csv(csv_filename, index=False)
            save_as_arff(csv_filename, arff_filename)
            
            print(f"Processado: {csv_filename} -> {arff_filename}")
        else:
            print("Nenhuma característica foi extraída. Verifique os caminhos das imagens.")

# Executa o processo
process_all_folders()
