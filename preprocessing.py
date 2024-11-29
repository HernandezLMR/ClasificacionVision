import os
from PIL import Image
from skimage import io, color
from skimage.feature import (
    canny,
    hog,
    local_binary_pattern,
    graycomatrix,
    graycoprops,
    corner_harris,
    corner_peaks,
    blob_log,
)
from numpy.fft import fft2, fftshift
import numpy as np
import pandas as pd
from tqdm import tqdm


def compress_image(image_path, output_size=(256, 256)):
    """Redimensionar y convertir imagen a escala de grises."""
    with Image.open(image_path) as img:
        img = img.convert("L")  # Convertir a escala de grises
        img = img.resize(output_size)  # Redimensionar la imagen
        return np.array(img)  # Devolver como arreglo de NumPy


def extract_features(image_array):
    """Extraer características de una imagen ya procesada."""
    gray_image = image_array

    # Características 1: Bordes (Canny)
    edges = canny(gray_image, sigma=1.0)
    canny_count = np.sum(edges)

    # Características 2: HOG (Histogram of Oriented Gradients)
    hog_features, _ = hog(
        gray_image,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm="L2-Hys",
    )
    hog_mean = np.mean(hog_features)

    # Características 3: LBP (Local Binary Patterns)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
    )
    lbp_hist_normalized = lbp_hist / np.sum(lbp_hist)

    # Características 4: GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(
        (gray_image * 255).astype(np.uint8),
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]

    # Características 5: Harris (Esquinas)
    corners = corner_harris(gray_image)
    harris_peaks = corner_peaks(corners, min_distance=5)
    harris_count = len(harris_peaks)

    # Características 6: Detección de blobs
    blobs = blob_log(gray_image, max_sigma=30, num_sigma=10, threshold=0.1)
    blob_count = len(blobs)

    # Características 7: FFT (Transformada de Fourier)
    fft_image = fftshift(fft2(gray_image))
    fft_energy = np.log(1 + np.abs(fft_image)).mean()

    # Características 8: BRIEF (Binary Robust Independent Elementary Features)
    from skimage.feature import BRIEF

    brief_extractor = BRIEF()
    brief_extractor.extract(gray_image, harris_peaks)
    brief_mean = (
        np.mean(brief_extractor.descriptors)
        if brief_extractor.descriptors is not None
        else 0
    )

    # Crear un vector de características
    feature_vector = np.array(
        [
            canny_count,
            hog_mean,
            *lbp_hist_normalized,
            contrast,
            homogeneity,
            correlation,
            energy,
            harris_count,
            blob_count,
            fft_energy,
            brief_mean,
        ]
    )
    return feature_vector


def process_parent_folder(parent_folder, output_size=(256, 256)):
    feature_matrix = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for class_folder in tqdm(os.listdir(parent_folder), desc="Processing folders"):
        folder_path = os.path.join(parent_folder, class_folder)
        if os.path.isdir(folder_path):  # Solo procesar carpetas
            image_files = [
                f
                for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
            for image_file in tqdm(
                image_files, desc=f"Processing images in {class_folder}", leave=False
            ):
                image_path = os.path.join(folder_path, image_file)
                try:
                    # Comprimir y procesar la imagen
                    compressed_image = compress_image(image_path, output_size)
                    features = extract_features(compressed_image)
                    # Agregar la clase como última columna
                    features_with_class = np.append(features, class_folder)
                    feature_matrix.append(features_with_class)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    return np.array(feature_matrix)


if __name__ == "__main__":
    parent_folder = "Wonders/Wonders of World"  # Cambia esta ruta al folder padre
    features = process_parent_folder(parent_folder, output_size=(256, 256))

    # Convertir a DataFrame y guardar como CSV
    columns = [f"feature_{i}" for i in range(features.shape[1] - 1)] + ["class"]
    df = pd.DataFrame(features, columns=columns)
    output_csv = "features_with_classes.csv"
    df.to_csv(output_csv, index=False)

    print(f"Features saved to {output_csv}")
