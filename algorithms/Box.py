import matplotlib.pyplot as plt
from PIL import Image

def box_filter(image, ksize):
    largura, altura = image.size
    saida = Image.new("L", (largura, altura))
    entrada = image.load()
    saidap = saida.load()    
    offset = ksize // 2

    for y in range(altura):
        for x in range(largura):
            total = 0
            aux = 0
            for dy in range(-offset, offset + 1):
                for dx in range(-offset, offset + 1):
                    ix = max(0, min(largura - 1, x + dx))
                    iy = max(0, min(altura - 1, y + dy))
                    total += entrada[ix, iy]
                    aux += 1
            saidap[x, y] = total // aux

    return saida

image = Image.open('shiva.jpg').convert("L")

kernel_sizes = [2, 3, 5, 7]
imagens = [box_filter(image, k) for k in kernel_sizes]

fig, resultados = plt.subplots(1, len(imagens) + 1, figsize=(15, 5))

resultados[0].imshow(image, cmap="gray")
resultados[0].set_title("Original")
resultados[0].axis("off")

for i, (img, k) in enumerate(zip(imagens, kernel_sizes), start=1):
    resultados[i].imshow(img, cmap="gray")
    resultados[i].set_title(f"Filtro {k}x{k}")
    resultados[i].axis("off")

plt.tight_layout()
plt.show()