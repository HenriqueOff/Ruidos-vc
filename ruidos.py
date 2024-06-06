from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Define valores padrão para os parâmetros
imagem = "exemplo.jpg"
tipo_ruido = "gauss"
parametro_ruido = 30
tipo_suavizacao = "gauss"
parametro_suavizacao = 10
tipo_borda = "sobel"
parametro_borda = 1000
interface_grafica = "sim"

# Mostra os parâmetros definidos
print("Executando sequência com")
print("Ruído = ", tipo_ruido)
print("Suavizador = ", tipo_suavizacao)
print("Imagem  = ", imagem)
print("Interface Gráfica = ", interface_grafica)

# Abre a imagem de entrada
original = io.imread(imagem)

def converter_para_escala_de_cinza(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    return imagem_cinza

# Função para adicionar ruído gaussiano à imagem
def adicionar_ruido_gaussiano(imagem, sigma=50):
    imagem_ruidosa = imagem.copy().astype(np.float64)  # Converter para float64
    ruido = np.random.normal(0, sigma, imagem.shape)
    imagem_ruidosa += ruido
    imagem_ruidosa = np.clip(imagem_ruidosa, 0, 255)  # Remover o limite de 1 para tipo de dado float
    imagem_ruidosa = imagem_ruidosa.astype(np.uint8)  # Converter de volta para uint8
    return imagem_ruidosa

# Função para adicionar ruído sal e pimenta à imagem
def adicionar_ruido_sal_e_pimenta(imagem, quantidade=0.05):
    imagem_ruidosa = imagem.copy()
    num_sal = np.ceil(quantidade * imagem.size * 0.5)
    num_pimenta = np.ceil(quantidade * imagem.size * 0.5)
    
    # Adiciona ruído "sal" (branco)
    coords_sal = [np.random.randint(0, i - 1, int(num_sal)) for i in imagem.shape]
    imagem_ruidosa[coords_sal[0], coords_sal[1]] = 255
    
    # Adiciona ruído "pimenta" (preto)
    coords_pimenta = [np.random.randint(0, i - 1, int(num_pimenta)) for i in imagem.shape]
    imagem_ruidosa[coords_pimenta[0], coords_pimenta[1]] = 0
    
    return imagem_ruidosa

# Função para adicionar ruído speckle à imagem
def adicionar_ruido_speckle(imagem, sigma=0.8):
    imagem_ruidosa = imagem.copy().astype(np.float64)
    ruido = np.random.normal(0, sigma, imagem.shape)
    imagem_ruidosa += imagem_ruidosa * ruido
    imagem_ruidosa = np.clip(imagem_ruidosa, 0, 255).astype(np.uint8)
    return imagem_ruidosa

# Aplica suavização media 
def aplicar_filtro_media(imagem, tamanho_kernel=parametro_suavizacao):
    imagem_filtrada = np.zeros_like(imagem)
    padding = tamanho_kernel // 2
    
    for i in range(padding, imagem.shape[0] - padding):
        for j in range(padding, imagem.shape[1] - padding):
            imagem_filtrada[i, j] = np.mean(imagem[i-padding:i+padding+1, j-padding:j+padding+1])
    
    return imagem_filtrada

#aplica suavizacao mediana
def aplicar_filtro_mediana(imagem, tamanho_kernel=parametro_suavizacao):
    imagem_suavizada = np.zeros_like(imagem)
    padding = tamanho_kernel // 2

    for i in range(padding, imagem.shape[0] - padding):
        for j in range(padding, imagem.shape[1] - padding):
            regiao = imagem[i-padding:i+padding+2, j-padding:j+padding+2]
            imagem_suavizada[i, j] = np.median(regiao)

    return imagem_suavizada

def difusao_anisotropica(imagem, iteracoes=50, delta_t=0.1, kappa=20):
    # Converte a imagem para escala de cinza se necessário
    if len(imagem.shape) == 3:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Cria uma cópia da imagem para aplicar a difusão
    imagem_difusa = np.float32(imagem.copy())

    for _ in range(iteracoes):
        # Calcula os gradientes da imagem
        gradiente_x = cv2.Sobel(imagem_difusa, cv2.CV_64F, 1, 0, ksize=3)
        gradiente_y = cv2.Sobel(imagem_difusa, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calcula o coeficiente de difusão
        coeficiente = 1 / (1 + ((gradiente_x**2 + gradiente_y**2) / kappa**2))
        
        # Atualiza os valores de intensidade da imagem
        imagem_difusa += delta_t * (
            coeficiente * (
                np.roll(imagem_difusa, -1, axis=0) +
                np.roll(imagem_difusa, 1, axis=0) +
                np.roll(imagem_difusa, -1, axis=1) +
                np.roll(imagem_difusa, 1, axis=1) -
                4 * imagem_difusa
            )
        )

    # Converte de volta para uint8
    imagem_difusa = np.clip(imagem_difusa, 0, 255).astype(np.uint8)

    return imagem_difusa


def deteccao_borda_sobel(imagem):
    # Convertendo a imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

    # Aplicar o operador Sobel
    gradiente_x = cv2.Sobel(imagem_cinza, cv2.CV_64F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(imagem_cinza, cv2.CV_64F, 0, 1, ksize=3)
    # Calcula a magnitude do gradiente
    magnitude_gradiente = np.sqrt(gradiente_x**2 + gradiente_y**2)
    magnitude_gradiente = np.uint8(magnitude_gradiente)
    return magnitude_gradiente


# Processa a imagem com os filtros
imagem_cinza_original = converter_para_escala_de_cinza(original)

imagem_com_ruido = adicionar_ruido_gaussiano(imagem_cinza_original)
imagem_com_ruido_sal_e_pimenta = adicionar_ruido_sal_e_pimenta(imagem_cinza_original)
imagem_com_ruido_speckle = adicionar_ruido_speckle(imagem_cinza_original)

imagem_suavizada = aplicar_filtro_media(original)
imagem_suavizada_mediana = aplicar_filtro_mediana(original)

imagem_suavizada_anisotropica = difusao_anisotropica(original)

bordas_detectadas_sobel = deteccao_borda_sobel(imagem_suavizada_mediana)


# Mostra as imagens resultantes
if interface_grafica == "sim":
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(original)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(imagem_com_ruido, cmap='gray') 
    ax[1].set_title('Ruído Gaussiano')

    ax[2].imshow(imagem_com_ruido_sal_e_pimenta,cmap='gray') 
    ax[2].set_title('Ruído Sal e Pimenta')

    ax[3].imshow(imagem_com_ruido_speckle, cmap='gray') 
    ax[3].set_title('Ruído Speckle')

    ax[4].imshow(imagem_suavizada, cmap='gray')
    ax[4].axis('off')
    ax[4].set_title('Suavizada por Media')

    ax[5].imshow(imagem_suavizada_mediana, cmap='gray')
    ax[5].axis('off')
    ax[5].set_title('Suavizada por Mediana')

    ax[6].imshow(imagem_suavizada_anisotropica, cmap='gray')
    ax[6].axis('off')
    ax[6].set_title('Difusao Anisotropica')

    ax[7].imshow(bordas_detectadas_sobel, cmap='gray')
    ax[7].axis('off')
    ax[7].set_title('Bordas Detectadas (Sobel)')


    fig.tight_layout()

    fig.canvas.manager.set_window_title('Resultados para imagem ' + imagem)
    plt.show()

# Salva no disco as imagens resultantes
io.imsave(imagem.split('.')[0] + "_ruido_gaussiano.jpg", imagem_com_ruido)
io.imsave(imagem.split('.')[0] + "_ruido_sal_e_pimenta.jpg", imagem_com_ruido_sal_e_pimenta)
io.imsave(imagem.split('.')[0] + "_ruido_speckle.jpg", imagem_com_ruido_speckle)
io.imsave(imagem.split('.')[0] + "_suavizada.jpg", imagem_suavizada)
io.imsave(imagem.split('.')[0] + "_suavizada_Mediana.jpg", imagem_suavizada_mediana)
io.imsave(imagem.split('.')[0] + "_bordas_detectadas_sobel.jpg", bordas_detectadas_sobel)
io.imsave(imagem.split('.')[0] + "_difusao_anisotropica.jpg", imagem_suavizada_anisotropica)
