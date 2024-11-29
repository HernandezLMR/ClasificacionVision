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
import os


def extract_features(image_path: str):
    image = io.imread(image_path)
    if len(image.shape) == 3:  # Convertir a escala de grises si es necesario
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image

    # Características como en el código original
    edges = canny(gray_image, sigma=1.0)
    canny_count = np.sum(edges)
    hog_features, _ = hog(
        gray_image,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm="L2-Hys",
    )
    hog_mean = np.mean(hog_features)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2)
    )
    lbp_hist_normalized = lbp_hist / np.sum(lbp_hist)
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
    corners = corner_harris(gray_image)
    harris_peaks = corner_peaks(corners, min_distance=5)
    harris_count = len(harris_peaks)
    blobs = blob_log(gray_image, max_sigma=30, num_sigma=10, threshold=0.1)
    blob_count = len(blobs)
    fft_image = fftshift(fft2(gray_image))
    fft_energy = np.log(1 + np.abs(fft_image)).mean()

    from skimage.feature import BRIEF

    brief_extractor = BRIEF()
    brief_extractor.extract(gray_image, harris_peaks)
    brief_mean = (
        np.mean(brief_extractor.descriptors)
        if brief_extractor.descriptors is not None
        else 0
    )

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


def process_folder(base_path):
    feature_matrix = []
    labels = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for class_folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, class_folder)
        if os.path.isdir(folder_path):
            for image_file in os.listdir(folder_path):
                if os.path.splitext(image_file)[1].lower() in valid_extensions:
                    image_path = os.path.join(folder_path, image_file)
                    try:
                        features = extract_features(image_path)
                        feature_matrix.append(features)
                        labels.append(class_folder)
                    except Exception as e:
                        with open("error_log.txt", "a") as log_file:
                            log_file.write(f"Error processing {image_path}: {e}\n")
    return np.array(feature_matrix), np.array(labels)


if __name__ == "__main__":
    base_path = "Wonders/Wonders of World"
    features, labels = process_folder(base_path)

    np.save("features.npy", features)
    np.save("labels.npy", labels)

    print("Features and labels saved successfully!")
