import os

# Caminho baseado na localização do script
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "teste.jpg")

print("Caminho absoluto gerado:", image_path)

# Tente carregar a imagem
import cv2
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Erro: A imagem não foi carregada corretamente.")
else:
    print("Imagem carregada com sucesso:", image.shape)

