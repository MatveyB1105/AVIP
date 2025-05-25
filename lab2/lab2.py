from PIL import Image
import numpy as np
from os import path

def prompt(variants: dict):
    for number, variant in enumerate(variants.keys(), 1):
        print(f'{number} - {variant}')
    input_correct = False
    user_input = 0
    while not input_correct:
        try:
            user_input = int(input('> '))
            if user_input <= 0 or user_input > len(variants):
                raise ValueError
            input_correct = True
        except ValueError:
            print("Введите корректное значение")
    return dict(enumerate(variants.values(), 1))[user_input]
def image_to_array(image_name: str) -> np.ndarray:
    file_path = path.join('pictures_src', image_name)
    if not path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    img_src = Image.open(file_path).convert('RGB')
    return np.array(img_src)

def save_image(image_array, filename):
    img = Image.fromarray(image_array.astype(np.uint8))
    img.save(path.join('pictures_results', filename))

def semitone(image_name):
    img = image_to_array(image_name)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    L = 0.3*R + 0.59*G + 0.11*B
    save_image((L).astype(np.uint8), f'Grayscale_{image_name}')

def adaptive_thresholding(image_name, window_size=7):
    grayscale_filename = path.join('pictures_results', f'Grayscale_{image_name}')

    grayscale_img = np.array(Image.open(grayscale_filename).convert('L'))

    height, width = grayscale_img.shape[:2]
    binary_img = np.zeros((height, width), dtype=np.uint8)

    pad = window_size // 2
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):

            window = grayscale_img[i - pad:i + pad + 1, j - pad:j + pad + 1]

            min_val = np.min(window).astype(np.int32)
            max_val = np.max(window).astype(np.int32)

            threshold = (min_val + max_val) / 2

            binary_img[i, j] = 255 if grayscale_img[i, j] > threshold else 0

    save_image(binary_img, f'Binary_{image_name}')

images = {
    "Example": 'example.png',
    "Chess": 'chess.png',
    "Text": 'text.png',
    "Nature": 'nature.png'
}

if __name__ == '__main__':
    print("Выберете изображение:")
    selected_image = prompt(images)
    print("Processing image:", selected_image)
    print("Converting to semitone...")
    semitone(selected_image)
    print("Applying adaptive thresholding...")
    adaptive_thresholding(selected_image)

