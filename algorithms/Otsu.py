import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def ler_imagem(caminho):
    imagem = Image.open(caminho).convert("L")  
    return np.array(imagem)  

def otsu_thresholding(imagem):
    histograma, _ = np.histogram(imagem, bins=256, range=(0, 256))  
    total_pixels = imagem.size

    Sum = np.sum([i * histograma[i] for i in range(256)])
    Sumb = 0  
    wb = 0  
    MaxVar = 128
    Limiar = 0  

    for i in range(MaxVar):
        wb += histograma[i]  
        if wb == 0:
            continue  

        wf = total_pixels - wb  
        if wf == 0:
            break  

        Sumb += i * histograma[i]  
        mb = Sumb / wb  
        mf = (Sum - Sumb) / wf  

        AVar = wb * wf * (mb - mf) ** 2  

        if AVar > MaxVar:
            MaxVar = AVar
            Limiar = i  

    return Limiar

imagem = ler_imagem('../images/moedas.jpg')

limiar = otsu_thresholding(imagem)
print(f"Limiar de Otsu encontrado: {limiar}")

imagem_segmentada = (imagem > limiar) * 255

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(imagem, cmap="gray")
plt.title("Imagem Original")

plt.subplot(1, 2, 2)
plt.imshow(imagem_segmentada, cmap="gray")
plt.title("Imagem Segmentada")

plt.show()
