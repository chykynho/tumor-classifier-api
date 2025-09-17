from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Carrega o modelo treinado
model = load_model('modelo_tumor.h5')

# ✅ CORRIGIDO: Lista de categorias com vírgulas corretas
categories = ['glioma', 'meningioma', 'notumor', 'pituitary']  # ← removido o [] errado em pituitary

# ✅ Descobre automaticamente o tamanho de entrada do modelo
IMG_HEIGHT, IMG_WIDTH = model.input_shape[1:3]  # Ex: (150, 150)
INPUT_CHANNELS = model.input_shape[3]  # 1 para escala de cinza, 3 para RGB

def preprocess_image(image_file):
    # Lê a imagem como bytes
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Imagem inválida ou corrompida")

    # ✅ Converte para escala de cinza se o modelo espera 1 canal
    if INPUT_CHANNELS == 1 and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif INPUT_CHANNELS == 3 and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ✅ Redimensiona para o tamanho que o modelo espera
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # ✅ Normaliza entre 0 e 1 (dividindo por 255, não 225!)
    img = img.astype('float32') / 255.0

    # ✅ Adiciona canal se necessário (para escala de cinza)
    if INPUT_CHANNELS == 1 and len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)  # (H, W) → (H, W, 1)

    # ✅ Adiciona dimensão de batch
    img = np.expand_dims(img, axis=0)  # (1, H, W, C)

    return img

@app.route('/')
def home():
    return "API de Classificação de Tumores Cerebrais - Envie uma imagem via POST para /predict"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Imagem sem nome'}), 400
    
    try:
        # Pré-processa a imagem
        img = preprocess_image(image_file)
        
        # Faz a predição
        predictions = model.predict(img)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = categories[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        # ✅ Retorna todas as probabilidades por categoria
        probabilities = {cat: float(prob) for cat, prob in zip(categories, predictions[0])}

        return jsonify({
            'classe_predita': predicted_class,
            'confianca': confidence,
            'probabilidades': probabilities
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # ← debug=False em produção!