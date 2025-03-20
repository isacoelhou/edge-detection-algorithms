import numpy as np
from math import sqrt, atan2
from PIL import Image
import cv2
import matplotlib.pyplot as plt

caminho_imagem = '../images/8.jpg'
 
def canny():
    matriz_imagem, largura, altura = ler_imagem_canny(caminho_imagem)

    matriz_imagem = np.array(matriz_imagem, dtype=np.float32)

    blurred_image = convolução(matriz_imagem, largura, altura, gaussian_kernel_canny(7, 1.0))
    magnitude, angulos = sobel_filter(blurred_image, largura, altura)
    bordas_finas = supressao_nao_maxima(magnitude, angulos)

    limiar_baixo = 30 
    limiar_alto = 100  
    bordas_finais = dupla_limiarizacao_conectividade(bordas_finas, limiar_baixo, limiar_alto)

    return bordas_finais

def Marr():

    sigma = 3.5
    threshold = 0.7
    imagem = cv2.imread(caminho_imagem, 0)

    g_kernel = gaussian_kernel(sigma)
    blurred = apply_convolution(imagem, g_kernel)
    laplacian = apply_convolution(blurred, laplacian_kernel)
    imagem_final = zero_crossing(laplacian, threshold)

    return imagem_final

def Otsu():
    imagem = Image.open(caminho_imagem).convert("L")  
    imagem = np.array(imagem)  
    limiar = otsu_thresholding(imagem)
    print(f"Limiar de Otsu encontrado: {limiar}")
    imagem_segmentada = (imagem > limiar) * 255
    
    return imagem_segmentada

def Watershed():
    imagem = canny()
    imagem = inundar(imagem, 1, 1, 0, 2)

    return imagem


def inundar(imagem, x, y, cor_alvo, nova_cor):
    # Se a cor inicial já for a nova cor, não há necessidade de preencher
    if imagem[x][y] != cor_alvo or imagem[x][y] == nova_cor:
        return

    # Criar uma pilha para armazenar os pixels a serem processados
    stack = [(x, y)]

    # Percorrer a pilha até que todos os pixels conectados sejam preenchidos
    while stack:
        cx, cy = stack.pop()  # Pegamos o último elemento da pilha

        # Se estiver dentro dos limites da imagem e for da cor alvo, preenchemos
        if 0 <= cx < len(imagem) and 0 <= cy < len(imagem[0]) and imagem[cx][cy] == cor_alvo:
            imagem[cx][cy] = nova_cor  # Preenche o pixel

            # Adicionamos os vizinhos na pilha para serem processados depois
            stack.append((cx + 1, cy))  # Direita
            stack.append((cx - 1, cy))  # Esquerda
            stack.append((cx, cy + 1))  # Baixo
            stack.append((cx, cy - 1))  # Cima


def ler_imagem_canny(caminho):
    imagem = Image.open(caminho).convert("L")  
    largura, altura = imagem.size
    matriz = [[0 for _ in range(largura)] for _ in range(altura)]
    
    for y in range(altura):
        for x in range(largura):
            matriz[y][x] = imagem.getpixel((x, y))  
    
    return matriz, largura, altura

def gaussian_kernel_canny(size, sigma):
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

def otsu_thresholding(imagem):
    histograma, _ = np.histogram(imagem, bins=255, range=(0, 255))  
    total_pixels = imagem.size

    Sum = np.sum([i * histograma[i] for i in range(255)])
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

# imagem_canny = canny()
# imagem_marr = Marr()
# imagem_otsu = Otsu()
imagem_water = Watershed()

imagem_water = np.array(imagem_water, dtype=np.uint8)
for i, linha in enumerate(imagem_water):
    for j, pixel in enumerate(linha):
        if pixel is None:
            print(f"Valor None encontrado em ({i}, {j})")


plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)  # Primeira posição
# plt.imshow(imagem_canny, cmap="gray")
# plt.title("Canny")

# plt.subplot(1, 3, 2)  # Segunda posição
# plt.imshow(imagem_marr, cmap="gray")
# plt.title("Marr-Hildreth")

plt.subplot(1, 3, 3)  # Terceira posição
plt.imshow(imagem_water, cmap="gray")
plt.title("Otsu")

plt.show()
