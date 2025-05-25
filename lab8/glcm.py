import numpy as np


def glcm(image, distance=1, angles=[0, 90, 180, 270], levels=256):
    glcms = []
    for angle in angles:
        glcm = np.zeros((levels, levels), dtype=np.float32)
        rows, cols = image.shape
        for i in range(rows):
            for j in range(cols):
                if angle == 0 and j + distance < cols:
                    glcm[image[i, j], image[i, j + distance]] += 1
                elif angle == 90 and i + distance < rows:
                    glcm[image[i, j], image[i + distance, j]] += 1
                elif angle == 180 and j - distance >= 0:
                    glcm[image[i, j], image[i, j - distance]] += 1
                elif angle == 270 and i - distance >= 0:
                    glcm[image[i, j], image[i - distance, j]] += 1


        if np.sum(glcm) != 0:
            glcms.append(glcm / np.sum(glcm))
        else:
            glcms.append(glcm)

    return glcms

def corr(glcm):
    corrs = []
    for matrix in glcm:
        levels = matrix.shape[0]
        i, j = np.meshgrid(range(levels), range(levels))
        mu_i = np.sum(i * matrix)
        mu_j = np.sum(j * matrix)
        sigma_i = np.sqrt(np.sum((i - mu_i)**2 * matrix))
        sigma_j = np.sqrt(np.sum((j - mu_j)**2 * matrix))
        corr = np.sum((i - mu_i) * (j - mu_j) * matrix) / (sigma_i * sigma_j)
        corrs.append(corr)
    return corrs


