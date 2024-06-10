from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pickle
import librosa
import numpy as np

# Load the trained model
model = load_model('singModel.keras')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def extract_features(file_path, n_mfcc=30, n_fft=1024):
    y, sr = librosa.load(file_path, sr=None)
    if len(y) < n_fft:
        y = np.pad(y, (0, max(0, n_fft - len(y))), mode='constant')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def predict_new_file(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)  # Reshape for prediction
    
    prediction = model.predict(features)
    
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    
    return predicted_label[0]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = f'/tmp/{file.filename}'
    file.save(file_path)

    predicted_label = predict_new_file(file_path)
    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
