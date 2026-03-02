import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def process_single_image(image_path):
    """
    Processa uma única imagem e retorna suas características extraídas
    (Haralick e LBP) em um vetor.

    Args:
        image_path (str): Caminho da imagem a ser processada.

    Returns:
        dict: Dicionário contendo as características extraídas.
    """
    def extract_haralick_features(image):
        """Extrai características de Haralick de uma imagem em escala de cinza."""
        glcm = graycomatrix(
            image,
            distances=[1, 3, 5],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            symmetric=True,
            normed=True
        )
        features = [graycoprops(glcm, prop).mean() for prop in ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM')]
        return features

    def extract_lbp_features(image, P=12, R=3):
        """Extrai características LBP (Local Binary Pattern) da imagem."""
        lbp = local_binary_pattern(image, P, R, method='uniform')
        n_bins = int(lbp.max() + 1)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return lbp_hist.tolist()

    # Ler imagem
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None or len(image.shape) != 2:
        raise ValueError("A imagem não foi carregada corretamente ou não é uma imagem em escala de cinza válida.")

    # Extrair características
    haralick_features = extract_haralick_features(image)
    lbp_features = extract_lbp_features(image)

    # Combinar todas as características
    all_features = haralick_features + lbp_features

    # Criar um dicionário de características com nomes apropriados
    feature_names = [f"haralick_{i}" for i in range(6)] + [f"lbp_{i}" for i in range(len(lbp_features))]
    feature_dict = dict(zip(feature_names, all_features))

    return feature_dict
