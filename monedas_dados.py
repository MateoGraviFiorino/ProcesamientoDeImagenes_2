import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para mostrar imágenes
def imshow(img, nueva_figura=True, titulo=None, img_a_color=False, bloqueante=True, barra_de_color=False, sin_ticks=False):
    print(img.shape)
    if nueva_figura:
        plt.figure()
    if img_a_color:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(titulo)
    if not sin_ticks:
        plt.xticks([]), plt.yticks([])
    if barra_de_color:
        plt.colorbar()
    if nueva_figura:
        plt.show(block=bloqueante)


imagen = cv2.imread("monedas.jpg")
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# kernel dilatación para cerrar los bordes
kernel_dilatacion = np.ones((10, 10), np.uint8)

# funcion para dilatar
def dilate_image(img, iterations=1):
    return cv2.dilate(img, kernel_dilatacion, iterations=iterations)

# Suavizar con Median/ probamos con Gaussian pero era mucho
imagen_suavizada = cv2.medianBlur(imagen_gris, 5)

# Usamos canny para los bordes
canny = cv2.Canny(imagen_suavizada, 50, 150)

# Dilatamos los bordes
bordes_dilatados = dilate_image(canny, iterations=3)

# Aplicamos erosión para eliminar pequeñas imperfecciones
img_er = cv2.erode(
    bordes_dilatados,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    iterations=1,
)

# Mostramos los bordes
imshow(img_er)

# Ahora si buscamos contornos
contornos, _ = cv2.findContours(img_er, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Contornos encontrados
len(contornos)

# Crear una máscara en blanco del mismo tamaño que la imagen original
mascara = np.zeros_like(imagen_gris)

# Dibujamos los contornos en la mascara
cv2.drawContours(mascara, contornos, -1, (255), thickness=cv2.FILLED)

# Encontramos componentes conectados en la máscara
(
    componentes_conectadas,
    etiquetas,
    estadisticas,
    centroides,
) = cv2.connectedComponentsWithStats(mascara, cv2.CV_32S, connectivity=8)

# Etiquetas y estadisticas 

"""
Columna 0 ([0]): Coordenada x del rectángulo delimitador superior izquierdo del componente.
Columna 1 ([1]): Coordenada y del rectángulo delimitador superior izquierdo del componente.
Columna 2 ([2]): Ancho del rectángulo delimitador del componente.
Columna 3 ([3]): Altura del rectángulo delimitador del componente.
Columna 4 ([4]): Área del componente conectado.
"""

imshow(etiquetas)
estadisticas


# Crear una imagen en blanco para el resultado
resultado = np.zeros_like(img_er)

# Filtrar y dibujar contornos basados en el área de los componentes conectados
for i in range(1, componentes_conectadas):
    area = estadisticas[i, cv2.CC_STAT_AREA]

    if area > 600:
        mascara = np.uint8(etiquetas == i)
        contorno, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(resultado, contorno, -1, 120, 2)

# Función para rellenar los contornos
def fill_contours(img):
    seed_point = (0, 0)
    blanco = (255, 255, 255)

    flags = 4
    lo_diff = (10, 10, 10)
    up_diff = (10, 10, 10)

    cv2.floodFill(img, None, seed_point, blanco, lo_diff, up_diff, flags)
    return ~img

# Rellenar contornos en la imagen resultante
resultado = fill_contours(resultado)

# Aplicar dilatación a la imagen resultante
result = cv2.dilate(
    resultado, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=5
)

# Encontrar componentes conectadas en la imagen resultante
(
    componentes_conectadas,
    etiquetas,
    estadisticas,
    centroides,
) = cv2.connectedComponentsWithStats(result, cv2.CV_32S, connectivity=8)

# Inicializar listas y variables para el procesamiento de cuadrados
squares_masks = []
aux = np.zeros_like(result)
labeled_image = cv2.merge([aux, aux, aux])
RHO_TH = 0.83

# Iterar sobre los componentes conectados y clasificar cuadrados
for i in range(1, componentes_conectadas):
    mascara = np.uint8(etiquetas == i)
    contorno, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = cv2.contourArea(contorno[0])
    perimetro = cv2.arcLength(contorno[0], True)
    rho = 4 * np.pi * area / (perimetro**2)
    flag_circ = rho > RHO_TH
    if flag_circ:
        if area > 123000:
            labeled_image[mascara == 1, 0] = 255
        elif area > 85000:
            labeled_image[mascara == 1, 1] = 255
        else:
            labeled_image[mascara == 1, 2] = 255
    else:
        labeled_image[mascara == 1, 2] = 120
        labeled_image[mascara == 1, 1] = 120

        square_mask = np.zeros_like(result)
        cv2.drawContours(square_mask, contorno, -1, 120, 2)
        squares_masks.append(square_mask)

# Combinar la imagen original con la imagen etiquetada
img_final = cv2.addWeighted(imagen, 0.7, labeled_image, 0.3, 0)

# Establecer un nuevo umbral para la circularidad de los cuadrados
RHO_TH = 0.78

# Iterar sobre las máscaras de los cuadrados y realizar operaciones de procesamiento
for id, square_mask in enumerate(squares_masks):
    square_mask = fill_contours(square_mask)
    squares = cv2.bitwise_and(canny, canny, mask=square_mask)

    # Aplicar dilatación a los cuadrados
    squares_dilatados = dilate_image(squares)

    # Encontrar componentes conectadas en los cuadrados
    (
        componentes_conectadas,
        etiquetas,
        estadisticas,
        centroides,
    ) = cv2.connectedComponentsWithStats(squares_dilatados, cv2.CV_32S, connectivity=8)
    
    count = 0 # Contador para numeros de dados
    for i in range(1, componentes_conectadas):
        area = estadisticas[i, cv2.CC_STAT_AREA]
        mascara = np.uint8(etiquetas == i)
        contorno, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filtrar cuadrados por área
        if area < 10000 and area > 500:
            area = cv2.contourArea(contorno[0])
            perimetro = cv2.arcLength(contorno[0], True)
            rho = 4 * np.pi * area / (perimetro**2)
            flag_circ = rho > RHO_TH
            
            # Contar cuadrados circulares
            if flag_circ:
                count += 1
    
    print(f"Dado {id + 1}, muestra el numero: {count}")

# Mostrar el resultado
imshow(img_final)