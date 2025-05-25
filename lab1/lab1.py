from PIL import Image
import numpy as np
from os import path

# Функция для преобразования изображения в NumPy массив
def image_to_array(image_name: str) -> np.ndarray:
    file_path = path.join('pictures_src', image_name)
    if not path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    img_src = Image.open(file_path).convert('RGB')
    return np.array(img_src)

# Функция для сохранения изображения
def save_image(image_array, filename):
    img = Image.fromarray(image_array.astype(np.uint8))
    img.save(path.join('pictures_results', filename))

# 1. Цветовые модели
# 1.1 Выделение компонентов R, G, B
def extract_rgb_components(image_name):
    img = image_to_array(image_name)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    save_image(np.stack([R, np.zeros_like(R), np.zeros_like(R)], axis=2), f'R_{image_name}')
    save_image(np.stack([np.zeros_like(G), G, np.zeros_like(G)], axis=2), f'G_{image_name}')
    save_image(np.stack([np.zeros_like(B), np.zeros_like(B), B], axis=2), f'B_{image_name}')

# 1.2 Преобразование в HSI и сохранение яркостной компоненты
def rgb_to_hsi(image_name):
    img = image_to_array(image_name).astype(np.float32) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    I = (R + G + B) / 3
    save_image((I * 255).astype(np.uint8), f'I_{image_name}')

# 1.3 Инвертирование яркостной компоненты
def invert_intensity(image_name):
    img = image_to_array(image_name).astype(np.float32) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    I = (R + G + B) / 3
    inverted_I = 1 - I
    factor = inverted_I / (I + 1e-10)
    R_new, G_new, B_new = R * factor, G * factor, B * factor
    inverted_img = np.stack([R_new, G_new, B_new], axis=2) * 255
    save_image(inverted_img.astype(np.uint8), f'Inverted_{image_name}')

# 2. Передискретизация
# Основная функция передискретизации
def one_step_resampling(img: np.ndarray, factor: float, f1, f2):
    dimensions = img.shape[0:2]
    new_dimensions = tuple(f1(dimension, factor) for dimension in dimensions)
    new_shape = (*new_dimensions, img.shape[2])
    new_img = np.empty(new_shape, dtype=np.uint8)
    for x in range(new_dimensions[0]):
        for y in range(new_dimensions[1]):
            new_img[x, y] = img[
                min(f2(x, factor), dimensions[0] - 1),
                min(f2(y, factor), dimensions[1] - 1)
            ]
    return new_img

# Растяжение (интерполяция) изображения в M раз
def stretch_image(image_name, M):
    img = image_to_array(image_name)
    stretched_img = one_step_resampling(img, M, lambda a, b: int(a * b), lambda a, b: int(round(a / b)))
    save_image(stretched_img, f'Stretched_M{M}_{image_name}')

# Сжатие (децимация) изображения в N раз
def compress_image(image_name, N):
    img = image_to_array(image_name)
    compressed_img = one_step_resampling(img, N, lambda a, b: int(round(a / b)), lambda a, b: int(a * b))
    save_image(compressed_img, f'Compressed_N{N}_{image_name}')

# Передискретизация в K = M/N раз за два прохода
def two_step_resampling(image_name, M, N):
    img = image_to_array(image_name)
    stretched_img = one_step_resampling(img, M, lambda a, b: int(a * b), lambda a, b: int(round(a / b)))
    resampled_img = one_step_resampling(stretched_img, N, lambda a, b: int(round(a / b)), lambda a, b: int(a * b))
    save_image(resampled_img, f'Resampled_K{M}_to_{N}_{image_name}')

# Передискретизация в K раз за один проход
def one_step_resampling_k(image_name, K):
    img = image_to_array(image_name)
    resampled_img = one_step_resampling(img, K, lambda a, b: int(round(a * b)), lambda a, b: int(round(a / b)))
    save_image(resampled_img, f'Resampled_K{K}_{image_name}')

# Главная функция для выполнения всех операций
def main(image_name):
    print("Processing image:", image_name)

    # 1. Цветовые модели
    print("Extracting RGB components...")
    extract_rgb_components(image_name)

    print("Converting to HSI and saving intensity component...")
    rgb_to_hsi(image_name)

    print("Inverting intensity component...")
    invert_intensity(image_name)

    # 2. Передискретизация
    M = 2  # Коэффициент растяжения
    N = 3  # Коэффициент сжатия
    K = M / N  # Общий коэффициент передискретизации

    print(f"Stretching image by factor {M}...")
    stretch_image(image_name, M)

    print(f"Compressing image by factor {N}...")
    compress_image(image_name, N)

    print(f"Resampling image in two steps (K = {M}/{N})...")
    two_step_resampling(image_name, M, N)

    print(f"Resampling image in one step (K = {K})...")
    one_step_resampling_k(image_name, K)

    print("All operations completed.")

if __name__ == "__main__":
    image_name = "cat.png"
    main(image_name)