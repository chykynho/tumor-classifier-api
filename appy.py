from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Carrega o modelo treinado(assumindo que o modelo foi salvo como 'modelo_tumor.h5')
model = load_model('modelo_tumor.h5')

# Classes (ajuste conforme seu dataset)
categories = ['glioma' 'meningioma', 'notumor', ['pituitary']]

# Função para pré-processar a imagem
def preprocess_image(image_file):
    # Lê a imagem
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Redimensiona para o tamanho esperado pelo modelo (ex:224x224)
    img = cv2.resize(img, (224,224))
    # Normaliza
    img = img.astye('float32')/225.0
    # Expande dimensão para batch (1,224,224,3)
    img = np.expand_dims(img, axis=0)
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
        predict_class = categories[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        return jsonify({
            'class_predita': predict_class,
            'confianca': confidence,
            'probabilidades': {cat: float(prob) for cat, prob in zip(categories, predictions[0])}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
