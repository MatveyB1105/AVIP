import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

image_name = "example.png"
output_dir = "pictures_results"
image_path = "pictures_src/Grayscale_example.png"
image = Image.open(image_path)

if image.mode == "RGB":
    image = image.convert("L")
gray = np.array(image)

Gx_kernel = np.array([[3, 0, -3],
                      [10, 0, -10],
                      [3, 0, -3]], dtype=np.float32) / 32

Gy_kernel = np.array([[3, 10, 3],
                      [0, 0, 0],
                      [-3, -10, -3]], dtype=np.float32) / 32

def convolve(image, kernel):
    rows, cols = image.shape
    k_rows, k_cols = kernel.shape
    pad_size = k_rows // 2
    padded_image = np.pad(image, pad_size, mode='constant')
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i + k_rows, j:j + k_cols]
            result[i, j] = np.sum(region * kernel)
    return result

Gx = convolve(gray, Gx_kernel)
Gy = convolve(gray, Gy_kernel)
G = np.sqrt(Gx ** 2 + Gy ** 2)

def normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return ((matrix - min_val) / (max_val - min_val) * 255).astype(np.uint8)


Gx_normalized = normalize(Gx)
Gy_normalized = normalize(Gy)
G_normalized = normalize(G)

threshold = 100
G_binary = (G_normalized > threshold).astype(np.uint8) * 255

def save_image(matrix, filename):
    result_image = Image.fromarray(matrix.astype(np.uint8))
    result_image.save(os.path.join(output_dir, filename))

save_image(gray, f'Gray_{image_name}')
save_image(Gx_normalized, f'Gx_{image_name}')
save_image(Gy_normalized, f'Gy_{image_name}')
save_image(G_normalized, f'G_{image_name}')
save_image(G_binary, f'G_norm_{image_name}')

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(gray, cmap="gray")
axes[0, 0].set_title("Полутоновое изображение")

axes[0, 1].imshow(Gx_normalized, cmap="gray")
axes[0, 1].set_title("Градиент Gx")

axes[0, 2].imshow(Gy_normalized, cmap="gray")
axes[0, 2].set_title("Градиент Gy")

axes[1, 0].imshow(G_normalized, cmap="gray")
axes[1, 0].set_title("Градиент G")

axes[1, 1].imshow(G_binary, cmap="gray")
axes[1, 1].set_title("Бинаризованная G")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()