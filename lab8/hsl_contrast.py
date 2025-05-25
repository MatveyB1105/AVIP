import numpy as np
import colorsys
from contrast import contrast

def hsl_contrast(img: np.array) -> np.array:
    hsl_img = np.zeros_like(img, dtype=float)
    height, width, _ = img.shape

    # Преобразование RGB → HSL
    for i in range(height):
        for j in range(width):
            r, g, b = img[i, j] / 255.0
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            hsl_img[i, j] = [h, l, s]

    # Извлекаем яркость и применяем contrast()
    lightness = (hsl_img[:, :, 1] * 255).astype(np.uint8)
    new_lightness = contrast(lightness) / 255.0  # Приводим обратно к диапазону [0,1]
    hsl_img[:, :, 1] = new_lightness

    # Преобразование HSL → RGB
    rgb_img = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            h, l, s = hsl_img[i, j]
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            rgb_img[i, j] = np.array([r, g, b]) * 255

    return rgb_img.astype(np.uint8)
