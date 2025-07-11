
import gradio as gr
import numpy as np
import joblib
import cv2
from PIL import Image
from skimage.feature import hog, graycomatrix, graycoprops

# Cargar modelo, scaler y codificador de etiquetas
model = joblib.load("models/fruit_classifier_multiclass.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

def extract_features(images):
    features = []
    for img in images:
        try:
            img_uint8 = (img * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)

            color_mean = np.mean(img, axis=(0, 1))
            color_std = np.std(img, axis=(0, 1))
            hsv_mean = np.mean(hsv, axis=(0, 1)) / 255.0
            lab_mean = np.mean(lab, axis=(0, 1)) / np.array([100, 256, 256])

            fd, _ = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, channel_axis=None)

            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            hist = hist.flatten() / hist.sum() if hist.sum() > 0 else hist.flatten()

            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                hu_moments = cv2.HuMoments(cv2.moments(cnt)).flatten()
            else:
                area = perimeter = circularity = 0
                hu_moments = np.zeros(7)

            glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]

            feature_vec = np.concatenate([
                color_mean, color_std, hsv_mean, lab_mean,
                fd, hist[:10],
                [area, perimeter, circularity],
                hu_moments[:5],
                [contrast, dissimilarity, homogeneity, energy, correlation]
            ])
            features.append(feature_vec)
        except Exception as e:
            print(f"Error al extraer características: {e}")
            features.append(np.zeros(50))
    return np.array(features)

def predict(image):
    image = image.resize((128, 128))
    img_np = np.array(image) / 255.0
    img_np = np.expand_dims(img_np, axis=0)

    features = extract_features(img_np)
    features_scaled = scaler.transform(features)
    pred_encoded = model.predict(features_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    estado, fruta = pred_label.split("_")
    estado_str = "Fresca ✅" if estado == "fresh" else "Podrida ❌"
    return f"Fruta detectada: {fruta.capitalize()}\nEstado: {estado_str}"

# Interfaz Gradio
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Sube una imagen de fruta"),
    outputs=gr.Textbox(label="Resultado de clasificación"),
    title="FruiGuard 🍎 - Clasificador de frutas y estado",
    description="Clasifica el tipo de fruta y si está fresca o podrida."
).launch()
