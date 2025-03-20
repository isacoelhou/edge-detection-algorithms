import matplotlib.pyplot as plt
from PIL import Image

def grayscale(value):
    if 0 <= value <= 50:
        return 25
    elif 51 <= value <= 100:
        return 75
    elif 101 <= value <= 150:
        return 125
    elif 151 <= value <= 200:
        return 175
    elif 201 <= value <= 255:
        return 255
    return value

def segmentacao(image):
    largura, altura = image.size
    imagem_segmentada = Image.new("L", (largura, altura))

    original = image.load()
    saida = imagem_segmentada.load()
    
    for y in range(altura):
        for x in range(largura):
            saida[x, y] = grayscale(original[x, y])
    
    return imagem_segmentada

caminho_imagem = "shiva.jpg"
img = Image.open(caminho_imagem).convert("L")
segmented = segmentacao(img)

fig, resultado = plt.subplots(1, 2, figsize=(10, 5))

resultado[0].imshow(img, cmap="gray", vmin=0, vmax=255)
resultado[0].set_title("Imagem Original")
resultado[0].axis("off")

resultado[1].imshow(segmented, cmap="gray", vmin=0, vmax=255)
resultado[1].set_title("Imagem Segmentada")
resultado[1].axis("off")

plt.show()
