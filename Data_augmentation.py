import os
import cv2
import numpy as np
from tqdm import tqdm
import random



def augment_image(image):
    """
    Aplica augmentações básicas em uma imagem.
    Args:
    - image (numpy array): Imagem original.
    Returns:
    - augmented_images (list): Lista de imagens aumentadas.
    """
    augmented_images = []
    
    # Flip horizontal
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    # Rotação
    rows, cols = image.shape[:2]
    for angle in [65, -65]:  # Rotacionar 65° e -65°
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        augmented_images.append(rotated)



   
    
    return augmented_images


def augment_images_in_folder(input_folder, output_folder, num_augmentations=10):
    """
    Realiza data augmentation em imagens de uma pasta e salva os resultados em outra.
    Args:
    - input_folder (str): Caminho para a pasta de entrada com imagens originais.
    - output_folder (str): Caminho para a pasta de saída com imagens aumentadas.
    - num_augmentations (int): Número de imagens aumentadas a serem geradas por imagem original.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Lista de arquivos na pasta de entrada
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc=f"Augmentando imagens em {input_folder}"):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        if image is not None:
            augmented_images = augment_image(image)

            # Salvar imagens aumentadas
            for i, aug_image in enumerate(augmented_images[:num_augmentations]):
                aug_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
                aug_path = os.path.join(output_folder, aug_filename)
                cv2.imwrite(aug_path, aug_image)
        else:
            print(f"Erro ao carregar a imagem {filename}. Ignorada.")

# Exemplo de uso
if __name__ == "__main__":
    # Definir pastas de entrada e saída
    input_folders = [
        'Dataset_full/Brain Tumor CT scan Images/Healthy',
        'Dataset_full/Brain Tumor CT scan Images/Tumor',
        'Dataset_full/Brain Tumor MRI images/Healthy',
        'Dataset_full/Brain Tumor MRI images/Tumor',
        'Dataset_full/glioma',
        'Dataset_full/meningioma',
        'Dataset_full/pituitary',
        'Dataset_full/notumor',
    ]
    
    output_folders = [
        'Dataset_full_Augmented/Brain Tumor_CT_scan_Images/Healthy',
        'Dataset_full_Augmented/Brain Tumor_CT_scan_Images/Tumor',
        'Dataset_full_Augmented/Brain_Tumor_MRI_images/Healthy',
        'Dataset_full_Augmented/Brain_Tumor_MRI_images/Tumor',
        'Dataset_full_Augmented/glioma',
        'Dataset_full_Augmented/meningioma',
        'Dataset_full_Augmented/pituitary',
        'Dataset_full_Augmented/brain-tumor-mri-dataset/notumor'
    ]

    # Realizar data augmentation para cada par de pastas
    for input_folder, output_folder in zip(input_folders, output_folders):
        augment_images_in_folder(input_folder, output_folder, num_augmentations=3)
