import numpy as np
from PIL import Image

def ler_imagem(caminho):
    imagem = Image.open(caminho).convert("L")  
    largura, altura = imagem.size
    matriz = np.array(imagem)  
    return matriz, largura, altura

def freeman(imagem, largura, altura):
    cadeia = []
    direcao_inicial = 0  
    x, y = None, None
    direcoes = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
    
    for k in range(altura):
        for i in range(largura):
            if imagem[k, i] == 255:  
                x, y = i, k
                break
        if x is not None:
            break  

    if x is None:
        return [], None  

    primeiro_ponto = (x, y)
    caminho = [(y, x)]

    while True:
        objeto = False
        for i in range(8):
            direcao_atual = (direcao_inicial + i) % 8  
            dx, dy = direcoes[direcao_atual]
            nx, ny = x + dx, y + dy

            if 0 <= nx < largura:
                if  0 <= ny < altura:
                    if  imagem[ny, nx] == 255:
                        cadeia.append(direcao_atual)
                        imagem[y, x] = 0  
                        x, y = nx, ny
                        caminho.append((y, x))
                        direcao_inicial = (direcao_atual + 6) % 8  
                        objeto = True
                        break

        if not objeto or (x, y) == primeiro_ponto:
            break 

    return cadeia, caminho

imagem, largura, altura = ler_imagem("../images/0.jpg")
imagem_freeman, caminho = freeman(imagem.copy(), largura, altura)

print("Cadeia de Freeman:", imagem_freeman)
print("Caminho:", caminho)