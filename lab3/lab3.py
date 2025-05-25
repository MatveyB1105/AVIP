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

def save_image(image_array, filename):
    img = Image.fromarray(image_array.astype(np.uint8))
    img.save(path.join('pictures_results', filename))

def image_to_array(image_name: str) -> np.ndarray:
    file_path = path.join('pictures_src', image_name)
    if not path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    img_src = Image.open(file_path).convert('RGB')
    return np.array(img_src)

def load_image(image_path):
    return np.array(Image.open(image_path).convert('L'))

def remove_white_fringe_binary_5x5(image_name):
    binary_filename = path.join('pictures_src', image_name)
    if not path.exists(binary_filename):
        raise FileNotFoundError(f"Binary image not found: {binary_filename}")

    binary_img = load_image(binary_filename)
    binary_img = (binary_img > 128).astype(np.uint8)

    height, width = binary_img.shape
    filtered_img = binary_img.copy()

    window_size = 5
    pad = window_size // 2

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            window = binary_img[i - pad:i + pad + 1, j - pad:j + pad + 1]

            if binary_img[i, j] == 1 and np.sum(window) <= 1:
                filtered_img[i, j] = 0
                continue

            if binary_img[i, j] == 1 and np.sum(window) < 9:
                filtered_img[i, j] = 0

    save_image(filtered_img * 255, f'Filtered_{image_name}')

    diff_img = np.bitwise_xor(binary_img, filtered_img) * 255
    save_image(diff_img, f'Diff_XOR_{image_name}')

images = {
    "Example": 'Binary_example.png',
    "Chess": 'Binary_chess.png',
    "Text": 'Binary_text.png',
    "Nature": 'Binary_nature.png'
}

if __name__ == '__main__':
    print("Выберете изображение:")
    selected_image = prompt(images)
    print("Processing image:", selected_image)
    print("Image filtering......")
    remove_white_fringe_binary_5x5(selected_image)