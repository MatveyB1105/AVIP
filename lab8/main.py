from my_io import prompt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from os import path, makedirs
from semitone import to_semitone
from glcm import glcm, corr
from contrast import contrast
from hsl_contrast import hsl_contrast

images = {
    'Bricks': 'kirp.png',
    'Pattern': 'oboi.png',
    'Wolf': 'wolf.png'
}

def estimate_run_length(matrix, percentile=95):

    run_lengths = np.nonzero(matrix)[1]
    if len(run_lengths) == 0:
        return 10  # fallback
    max_length = int(np.percentile(run_lengths, percentile))
    return max(5, min(max_length + 1, matrix.shape[1]))


def visualize_glcm(matrix, title, save_path):
    matrix = matrix + 1e-6

    log_scaled = np.log1p(matrix)

    # Нормализация
    norm = log_scaled / np.max(log_scaled) if np.max(log_scaled) != 0 else log_scaled
    norm *= 255

    plt.figure(figsize=(8, 6))
    plt.imshow(norm.astype(np.uint8), cmap='gray', aspect='equal')
    plt.title(title)
    plt.xlabel('Яркость пикселя j')
    plt.ylabel('Яркость пикселя i')
    plt.colorbar(label='log(1 + значение)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def ensure_dirs():
    for folder in ['results/semitone', 'results/contrasted', 'results/histograms',
                   'results/glcm', 'results/glcm_contrasted']:
        makedirs(folder, exist_ok=True)

if __name__ == '__main__':
    ensure_dirs()
    print('Выберите изображение:')
    selected_image = prompt(images)

    semitone_img = to_semitone(selected_image)
    semitone_path = path.join('results', 'semitone', selected_image)
    semitone_img.save(semitone_path)

    semi = np.array(Image.open(semitone_path).convert('L'))
    transformed = contrast(semi)
    transformed_path = path.join('results', 'contrasted', selected_image)
    Image.fromarray(transformed.astype(np.uint8), "L").save(transformed_path)

    original_rgb = np.array(Image.open(path.join('src', selected_image)).convert('RGB'))
    contrasted_color = hsl_contrast(original_rgb)
    color_transformed_path = path.join('results', 'contrasted', f"color_{selected_image}")
    Image.fromarray(contrasted_color).save(color_transformed_path)

    figure, axis = plt.subplots(2, 1, figsize=(6, 6))
    axis[0].hist(x=semi.flatten(), bins=np.arange(1, 255))
    axis[0].set_title('Исходное изображение')

    axis[1].hist(x=transformed.flatten(), bins=np.arange(1, 255))
    axis[1].set_title('Преобразованное изображение')
    plt.tight_layout()
    hist_path = path.join('results', 'histograms', selected_image)
    plt.savefig(hist_path)
    plt.close()

    angles = [0, 90, 180, 270]
    for angle in angles:
        print(f'\n--- Направление {angle}° ---')

        # Исходное изображение
        glcm_matrix = glcm(semi.astype(np.uint8), distance=1, angles=[angle], levels=256)[0]
        corr_value = corr([glcm_matrix])[0]

        visualize_glcm(
            glcm_matrix,
            title=f"GLCM {angle}° — исходное",
            save_path=path.join('results', 'glcm', f'{angle}_{selected_image}')
        )

        t_glcm_matrix = glcm(transformed.astype(np.uint8), distance=1, angles=[angle], levels=256)[0]
        t_corr_value = corr([t_glcm_matrix])[0]

        visualize_glcm(
            t_glcm_matrix,
            title=f"GLCM {angle}° — контрастированное",
            save_path=path.join('results', 'glcm_contrasted', f'{angle}_{selected_image}')
        )

        print(f"CORR: {corr_value:.2f}")
        print(f"CORR (contrasted): {t_corr_value:.2f}")
