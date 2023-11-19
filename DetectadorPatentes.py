import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def imshow(img, new_fig=True, title=None, color_img=False, blocking=True, colorbar=False, ticks=False):
    print(img.shape)
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

def gray_scale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imshow(gray,new_fig=True, title="Imagen con escalado de grises")
    return gray

def adaptive_threshold(img):
    tresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, -10)
    imshow(tresh, new_fig=True, title="Imagen con umbralado adaptativo")
    return tresh

def connected_components(img_tresh, min_area, max_area, min_aspect_ratio, max_aspect_ratio):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_tresh, connectivity=4)

    filtered_area = np.zeros_like(img_tresh)

    for i in range(1, num_labels):
        aspect_ratio = stats[i, cv2.CC_STAT_WIDTH] / stats[i, cv2.CC_STAT_HEIGHT]

        if (min_area < stats[i][4] and stats[i][4] < max_area and min_aspect_ratio < aspect_ratio and aspect_ratio < max_aspect_ratio):
            filtered_area[labels == i] = img_tresh[labels == i]

    return filtered_area

def remove_noise(image_bin, distance_threshold):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_bin, connectivity=4)

    for i in range(1, num_labels):
        centroid = centroids[i]
        distances = np.linalg.norm(centroid - centroids, axis=1)
        distances[i] = np.inf
        min_distance = min(distances)

        if min_distance > distance_threshold:
            image_bin[labels == i] = 0

    return image_bin

def remove_noise_and_smooth(filtered_area, distance_threshold):
    clean_area = remove_noise(filtered_area.copy(), distance_threshold)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    clean_area_open = cv2.morphologyEx(clean_area, cv2.MORPH_OPEN, se)
    final_image = cv2.morphologyEx(clean_area_open, cv2.MORPH_CLOSE, se)
    imshow(final_image , new_fig=True, title="Imagen con filtrado final")

    return final_image

def extract_characters(final_image):
    w, h = final_image.shape
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(final_image[80:w-50, 170:h-50], connectivity=4)

    for label in range(1, num_labels):  
        component_mask = (labels == label).astype(np.uint8) * 255
        imshow(component_mask , new_fig=True, title="Componentes")

# Uso
img_path = os.path.join("Patentes", "img01.png")

if os.path.isfile(img_path):
    img_original = cv2.imread(img_path)
    img_gray = gray_scale(img_original)
    img_thresh = adaptive_threshold(img_gray)

    # Ajusta estos parámetros según tus necesidades
    min_area, max_area = 25, 90
    min_aspect_ratio, max_aspect_ratio = 0.4, 0.7
    filtered_area = connected_components(img_thresh, min_area, max_area, min_aspect_ratio, max_aspect_ratio)

    # Ajusta este parámetro según tus necesidades
    distance_threshold = 15
    img_final = remove_noise_and_smooth(filtered_area, distance_threshold)

    extract_characters(img_final)
else:
    print("La ruta de la imagen no es válida.")
