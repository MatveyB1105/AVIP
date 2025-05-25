import numpy as np

def contrast(img: np.array, low_perc=2, high_perc=98):
    img = img.astype(np.float32)


    low = np.percentile(img, low_perc)
    high = np.percentile(img, high_perc)

    if high - low == 0:
        return np.zeros_like(img)

    contrasted = (img - low) * (255.0 / (high - low))
    return np.clip(contrasted, 0, 255)


    
    
    

   