import numpy as np
from math import sqrt, atan2
from PIL import Image

def ler_imagem(caminho):
    imagem = Image.open(caminho).convert("L")  
    largura, altura = imagem.size
    matriz = [[0 for _ in range(largura)] for _ in range(altura)]
    
    for y in range(altura):
        for x in range(largura):
            matriz[y][x] = imagem.getpixel((x, y))  
    
    return matriz, largura, altura

def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    k = size // 2
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            kernel[i+k, j+k] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(i**2 + j**2) / (2 * sigma**2))

    soma_total = 0.0
    for i in range(size):
        for j in range(size):
            soma_total += kernel[i, j] 

    for i in range(size):
        for j in range(size):
            kernel[i, j] /= soma_total

    return kernel

def convolução(image, largura, altura, kernel):
    kernel_altura, kernel_largura = kernel.shape
    pad_altura = 10 * kernel_altura // 2
    pad_largura = 10 * kernel_largura // 2

    padded_image = np.zeros((altura + 2*pad_altura, largura + 2*pad_largura))
    for y in range(altura):
        for x in range(largura):
            padded_image[y+pad_altura, x+pad_largura] = image[y][x]

    output = np.zeros_like(image, dtype=np.float32)

    for i in range(altura):
        for j in range(largura):
            soma = 0.0
            for ki in range(kernel_altura):
                for kj in range(kernel_largura):
                    soma += padded_image[i + ki, j + kj] * kernel[ki, kj]
            output[i, j] = soma

    return output

def sobel_filter(image, largura, altura):
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])

    altura = len(image)
    largura = len(image[0])

    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)

    for y in range(1, altura - 1):  
        for x in range(1, largura - 1):
            gx[y][x] = (kernel_x[0][0] * image[y-1][x-1] + kernel_x[0][1] * image[y-1][x] + kernel_x[0][2] * image[y-1][x+1] +
                       kernel_x[1][0] * image[y][x-1]   + kernel_x[1][1] * image[y][x]   + kernel_x[1][2] * image[y][x+1] +
                       kernel_x[2][0] * image[y+1][x-1] + kernel_x[2][1] * image[y+1][x] + kernel_x[2][2] * image[y+1][x+1])

            gy[y][x] = (kernel_y[0][0] * image[y-1][x-1] + kernel_y[0][1] * image[y-1][x] + kernel_y[0][2] * image[y-1][x+1] +
                       kernel_y[1][0] * image[y][x-1]   + kernel_y[1][1] * image[y][x]   + kernel_y[1][2] * image[y][x+1] +
                       kernel_y[2][0] * image[y+1][x-1] + kernel_y[2][1] * image[y+1][x] + kernel_y[2][2] * image[y+1][x+1])

    magnitude = np.zeros_like(image, dtype=np.float32)
    for y in range(altura):
        for x in range(largura):
            magnitude[y][x] = sqrt(gx[y][x]**2 + gy[y][x]**2)

    angulos = np.zeros_like(image, dtype=np.float32)
    for y in range(altura):
        for x in range(largura):
            angulos[y][x] = atan2(gy[y][x], gx[y][x]) 

    return magnitude, angulos

def supressao_nao_maxima(magnitude, angulos):

    altura, largura = magnitude.shape
    supressao = np.zeros_like(magnitude, dtype=np.float32)

    angulos = np.rad2deg(angulos) % 180

    for y in range(1, altura - 1):
        for x in range(1, largura - 1):
            direcao = angulos[y][x]

            if (0 <= direcao < 22.5) or (157.5 <= direcao <= 180):
                vizinho1 = magnitude[y][x + 1]
                vizinho2 = magnitude[y][x - 1]
            elif 22.5 <= direcao < 67.5:
                vizinho1 = magnitude[y + 1][x - 1]
                vizinho2 = magnitude[y - 1][x + 1]
            elif 67.5 <= direcao < 112.5:
                vizinho1 = magnitude[y + 1][x]
                vizinho2 = magnitude[y - 1][x]
            elif 112.5 <= direcao < 157.5:
                vizinho1 = magnitude[y - 1][x - 1]
                vizinho2 = magnitude[y + 1][x + 1]

            if magnitude[y][x] >= vizinho1 and magnitude[y][x] >= vizinho2:
                supressao[y][x] = magnitude[y][x]
            else:
                supressao[y][x] = 0

    return supressao

def dupla_limiarizacao_conectividade(magnitude, limiar_baixo, limiar_alto):

    altura, largura = magnitude.shape
    bordas_fortes = np.zeros_like(magnitude, dtype=np.uint8)
    bordas_fracas = np.zeros_like(magnitude, dtype=np.uint8)

    for y in range(altura):
        for x in range(largura):
            valor = magnitude[y][x]
            if valor >= limiar_alto:
                bordas_fortes[y][x] = 255 
            elif valor >= limiar_baixo:
                bordas_fracas[y][x] = 255  

    for y in range(1, altura - 1):
        for x in range(1, largura - 1):
            if bordas_fracas[y][x] == 255: 
                found = False
                for dy in range(-1, 2):  
                    for dx in range(-1, 2): 
                        if bordas_fortes[y + dy][x + dx] == 255:
                            found = True
                            break
                    if found:
                        break
                if found:
                    bordas_fortes[y][x] = 255

    return bordas_fortes


caminho_imagem = "images/0.jpg" 
matriz_imagem, largura, altura = ler_imagem(caminho_imagem)

matriz_imagem = np.array(matriz_imagem, dtype=np.float32)

blurred_image = convolução(matriz_imagem, largura, altura, gaussian_kernel(7, 1.0))
magnitude, angulos = sobel_filter(blurred_image, largura, altura)
bordas_finas = supressao_nao_maxima(magnitude, angulos)

limiar_baixo = 30 
limiar_alto = 100  
bordas_finais = dupla_limiarizacao_conectividade(bordas_finas, limiar_baixo, limiar_alto)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Imagem Original')
plt.imshow(matriz_imagem, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Bordas Detectadas')
plt.imshow(bordas_finais, cmap='gray')

plt.show()