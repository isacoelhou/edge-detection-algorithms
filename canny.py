import numpy as np
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
            kernel[i+k, j+k] = np.exp(-(i**2 + j**2) / (2 * sigma**2))

    kernel /= kernel.sum()  
    return kernel

def convolve2d(image, kernel):
    image_height = len(image)
    image_width = len(image[0])
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.zeros((image_height + 2*pad_height, image_width + 2*pad_width))
    for y in range(image_height):
        for x in range(image_width):
            padded_image[y+pad_height, x+pad_width] = image[y][x]

    output = np.zeros_like(image, dtype=np.float32)

    for i in range(image_height):
        for j in range(image_width):
            output[i][j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output


import numpy as np
from math import sqrt
from math import sqrt, atan2, degrees


def sobel_filter(image, largura, altura):
    """Aplica o filtro de Sobel para detectar bordas em uma imagem."""
    # Kernels de Sobel
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])

    # Inicializa as matrizes para os gradientes
    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)

    # Aplica a convolução com os kernels de Sobel
    for y in range(1, altura - 1):  # Ignora as bordas da imagem
        for x in range(1, largura - 1):
            gx[y][x] = (kernel_x[0][0] * image[y-1][x-1] + kernel_x[0][1] * image[y-1][x] + kernel_x[0][2] * image[y-1][x+1] +
                       kernel_x[1][0] * image[y][x-1]   + kernel_x[1][1] * image[y][x]   + kernel_x[1][2] * image[y][x+1] +
                       kernel_x[2][0] * image[y+1][x-1] + kernel_x[2][1] * image[y+1][x] + kernel_x[2][2] * image[y+1][x+1])

            # Gradiente vertical (Gy)
            gy[y][x] = (kernel_y[0][0] * image[y-1][x-1] + kernel_y[0][1] * image[y-1][x] + kernel_y[0][2] * image[y-1][x+1] +
                       kernel_y[1][0] * image[y][x-1]   + kernel_y[1][1] * image[y][x]   + kernel_y[1][2] * image[y][x+1] +
                       kernel_y[2][0] * image[y+1][x-1] + kernel_y[2][1] * image[y+1][x] + kernel_y[2][2] * image[y+1][x+1])

    # Calcula a magnitude do gradiente
    magnitude = np.zeros_like(image, dtype=np.float32)
    for y in range(altura):
        for x in range(largura):
            magnitude[y][x] = sqrt(gx[y][x]**2 + gy[y][x]**2)

    # Normaliza a magnitude para o intervalo [0, 255]
    magnitude_normalizada = (magnitude / magnitude.max()) * 255

    angulos = np.zeros_like(image, dtype=np.float32)
    for y in range(altura):
        for x in range(largura):
            angulos[y][x] = atan2(gy[y][x], gx[y][x])  # Ângulo em radianos

    return magnitude, angulos


def supressao_nao_maxima(magnitude, angulos):
    """
    Aplica a supressão não máxima na matriz de magnitude do gradiente.
    
    Parâmetros:
        magnitude: Matriz de magnitude do gradiente (saída do filtro de Sobel).
        angulos: Matriz de ângulos do gradiente (direção do gradiente).
    
    Retorna:
        Matriz com as bordas afinadas após a supressão não máxima.
    """
    altura, largura = magnitude.shape
    supressao = np.zeros_like(magnitude, dtype=np.float32)

    # Converte os ângulos para graus (0 a 180)
    angulos = np.rad2deg(angulos) % 180

    for y in range(1, altura - 1):
        for x in range(1, largura - 1):
            direcao = angulos[y][x]

            # Determina os pixels vizinhos na direção do gradiente
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

            # Supressão não máxima: mantém apenas os máximos locais
            if magnitude[y][x] >= vizinho1 and magnitude[y][x] >= vizinho2:
                supressao[y][x] = magnitude[y][x]
            else:
                supressao[y][x] = 0

    return supressao

def dupla_limiarizacao_conectividade(magnitude, limiar_baixo, limiar_alto):
    altura, largura = magnitude.shape
    bordas_fortes = np.zeros_like(magnitude, dtype=np.uint8)
    bordas_fracas = np.zeros_like(magnitude, dtype=np.uint8)

    # Dupla limiarização
    for y in range(altura):
        for x in range(largura):
            valor = magnitude[y][x]
            if valor >= limiar_alto:
                bordas_fortes[y][x] = 255  # Borda forte
            elif valor >= limiar_baixo:
                bordas_fracas[y][x] = 255  # Borda fraca

    # Análise de conectividade com vizinhança 5x5
    for y in range(2, altura - 2):
        for x in range(2, largura - 2):
            if bordas_fracas[y][x] == 255:  # Se for uma borda fraca
                # Verifica se está conectada a uma borda forte na vizinhança 5x5
                if np.any(bordas_fortes[y-2:y+3, x-2:x+3] == 255):
                    bordas_fortes[y][x] = 255  # Converte borda fraca em forte

    return bordas_fortes

def salva_imagem(matriz, largura, altura, caminho_saida):
    from PIL import Image

    imagem = Image.new("L", (largura, altura))
    
    for y in range(altura):
        for x in range(largura):
            valor = int(matriz[y][x])
            valor = max(0, min(valor, 255)) 
            imagem.putpixel((x, y), valor)
    
    imagem.save(caminho_saida)


matriz_imagem, largura, altura = ler_imagem( "./images/1.jpg"  )
matriz_imagem = np.array(matriz_imagem, dtype=np.float32)

blurred_image = convolve2d(matriz_imagem, gaussian_kernel(7, 1.5))
sobel_image, angulos = sobel_filter(blurred_image, largura, altura)
nms_image = supressao_nao_maxima(sobel_image, angulos)
limiar_baixo = 50  # Defina o limiar baixo
limiar_alto = 100  # Defina o limiar alto
bordas_finais = dupla_limiarizacao_conectividade(nms_image, limiar_baixo, limiar_alto)

salva_imagem(bordas_finais, largura, altura ,"out.jpg")