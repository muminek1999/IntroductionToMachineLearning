import os
import sys
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import img_as_ubyte


def get_color_features(image: np.ndarray, bins: tuple = (8, 8, 8)) -> np.ndarray:
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)

    return hist.flatten()

def get_haralick_features(image: np.ndarray) -> np.ndarray:
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img_ubyte = img_as_ubyte(gray_img)

    glcm = graycomatrix(gray_img_ubyte,
                        distances=[1],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256,
                        symmetric=True,
                        normed=True)

    features = []
    props = ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity']
    for prop in props:
        features.append(np.mean(graycoprops(glcm, prop)))

    return np.array(features)

def get_lbp_features(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')

    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-10)

    return hist

def get_shape_features(image: np.ndarray) -> np.ndarray:
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return np.zeros(7)

    main_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(main_contour)
    hu_moments = cv2.HuMoments(moments)
    hu_moments_stable = np.where(hu_moments == 0, 1e-10, hu_moments)
    log_hu_moments = -np.sign(hu_moments_stable) * np.log(np.abs(hu_moments_stable))

    return log_hu_moments.flatten()

# Funkcja do normalizacji - euclidian normalization

def normalize_features(feature_vector: np.ndarray) -> np.ndarray:
  
    norm = np.linalg.norm(feature_vector)
    
    if norm == 0:
        return feature_vector 
        
    return feature_vector / norm


def extract_features(processed_image: np.ndarray) -> np.ndarray:
    try:
        if processed_image is None:
            raise ValueError("Incorrect processed image")

        color_features = get_color_features(processed_image)
        haralick_features = get_haralick_features(processed_image)
        lbp_features = get_lbp_features(processed_image)
        shape_features = get_shape_features(processed_image)

        final_feature_vector = np.concatenate([
            color_features,
            haralick_features,
            lbp_features,
            shape_features
        ])

        normalized_vector = normalize_features(final_feature_vector)

        return normalized_vector

    except Exception as e:
        print(f"ERROR: An unexpected error occurred during feature extraction: {e}")
        sys.exit(1)