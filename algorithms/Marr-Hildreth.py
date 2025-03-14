import cv2
import numpy as np
import matplotlib.pyplot as plt

laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

def apply_convolution(imagem, filtro_de_kernel):
    altura_kernel, largura_kernel = filtro_de_kernel.shape
    altura_imagem, largura_imagem = imagem.shape

    padded_image = np.pad(imagem, ((altura_kernel // 2, largura_kernel // 2)), mode="constant")

    resultado = np.empty_like(imagem, dtype=np.float32)

    for linha in range(altura_imagem):
        for col in range(largura_imagem):
            regiao = padded_image[linha : linha + altura_kernel, col : col + largura_kernel]
            resultado[linha][col] = np.sum(regiao * filtro_de_kernel)

    return resultado

import numpy as np

def gaussian_kernel(sigma):
    tamanho = int(3 * sigma)
    kernel_size = 2 * tamanho + 1
    
    kernel = [[0.0 for _ in range(kernel_size)] for _ in range(kernel_size)]
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - tamanho
            y = j - tamanho
            
            kernel[i][j] = (2.71828 ** (-(x**2 + y**2) / (2 * sigma**2))) / (2 * 3.14159 * sigma**2)
    
    soma = 0.0
    for linha in kernel:
        for valor in linha:
            soma += valor
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] = kernel[i][j] / soma
    
    return np.array(kernel, dtype=np.float32)

def zero_crossing(imagem, threshold):

    N, M = imagem.shape
    bordas = np.zeros_like(imagem, dtype=np.uint8)

    for i in range(N):
        for j in range(M):
            if i > 0 and i < N - 1:
                esquerda = imagem[i - 1, j]
                direita = imagem[i + 1, j]

                if esquerda * direita < 0 and np.abs(esquerda - direita) > threshold:
                    bordas[i, j] = 255

            if j > 0 and j < M - 1:
                cima = imagem[i, j + 1]
                baixo = imagem[i, j - 1]

                if cima * baixo < 0 and np.abs(cima - baixo) > threshold:
                    bordas[i, j] = 255

            if (i > 0 and i < N - 1) and (j > 0 and j < M - 1):
                cima_esquerda = imagem[i - 1, j - 1]
                baixo_direita = imagem[i + 1, j + 1]
                baixo_esquerda = imagem[i - 1, j + 1]
                cima_direita = imagem[i + 1, j - 1]

                if (
                    cima_esquerda * baixo_direita < 0
                    and np.abs(cima_esquerda - baixo_direita) > threshold
                ):
                    bordas[i, j] = 255

                elif (
                    baixo_esquerda * cima_direita < 0
                    and np.abs(baixo_esquerda - cima_direita) > threshold
                ):
                    bordas[i, j] = 255

    return bordas

def LerImagen(caminho):

    from PIL import Image
    imagem = Image.open(caminho).convert("L")
    largura, altura = imagem.size
    matriz = [[0 for _ in range(largura)] for _ in range(altura)]
    
    for y in range(altura):
        for x in range(largura):
            matriz[y][x] = 1 if imagem.getpixel((x, y)) > 128 else 0
    
    return matriz, largura, altura

sigma = 3.5
threshold = 0.7
imagem = cv2.imread("./images/0.jpg", 0)

g_kernel = gaussian_kernel(sigma)
blurred = apply_convolution(imagem, g_kernel)
laplacian = apply_convolution(blurred, laplacian_kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Imagem Original')
plt.imshow(imagem)

plt.subplot(1, 2, 2)
plt.title('Bordas Detectadas')
plt.imshow(zero_crossing(laplacian, threshold))

plt.show()