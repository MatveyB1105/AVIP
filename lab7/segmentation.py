import numpy as np
import matplotlib.pyplot as plt
from os import path
from helpers import calculate_profile
from PIL import Image, ImageDraw
from PIL.ImageOps import invert

def split_letters(img: np.array, profile: np.array):
    assert img.shape[1] == profile.shape[0]
    letters = []
    letter_borders = []
    letter_start = 0
    is_empty = True

    for i in range(img.shape[1]):
        if profile[i] == 0:
            if not is_empty:
                is_empty = True
                letters.append(img[:, letter_start:i + 1])
                letter_borders.append(i+1)

        else:
            if is_empty:
                is_empty = False
                letter_start = i
                letter_borders.append(letter_start)

    letters.append(img[:, letter_start:img.shape[1] - 1])

    return letters, letter_borders


def bar(data, bins, axis):
    if axis == 1:
        plt.bar(x=bins, height=data)

    elif axis == 0:
        plt.barh(y=bins, width=data)

    else:
        raise ValueError('Invalid axis')

if __name__ == '__main__':
    img = np.array(Image.open('src/sentence_white2.bmp').convert('L'))

    profile_x = calculate_profile(img, 0)
    profile_y = calculate_profile(img, 1)

    img_letters, letter_borders = split_letters(img, profile_y)

    result_img = Image.fromarray(img.astype(np.uint8), 'L')
    rgb_img = Image.new("RGB", result_img.size)
    rgb_img.paste(result_img)
    draw = ImageDraw.Draw(rgb_img)

    for i, letter in enumerate(img_letters):
        for axis in (0, 1):
            letter_img = Image.fromarray(letter.astype(np.uint8), 'L').convert('1')

        letter_img = invert(letter_img)
        letter_img.save(f"results/symbols/letter_{i}.png")