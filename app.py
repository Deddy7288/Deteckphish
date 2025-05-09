import os
import joblib # Anda sepertinya menggunakan joblib untuk scaler, pastikan ini benar
from flask import Flask, request, jsonify
from flask_cors import CORS # <<<--- 1. Import CORS
from urllib.parse import urlparse
import numpy as np
import re
import tldextract
# import pickle # Komentari atau hapus jika tidak memakai pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler # Import StandardScaler (meskipun scaler dimuat, kadang diperlukan)

# Inisialisasi Aplikasi Flask
app = Flask(__name__)
CORS(app) # <<<--- 2. Aktifkan CORS untuk semua route dan origin (aman untuk development)
          # Untuk production, pertimbangkan konfigurasi yang lebih ketat, misal:
          # CORS(app, resources={r"/predict": {"origins": "http://domain-frontend-anda.com"}})

# --- Memuat Model dan Scaler ---
# Lakukan ini di luar fungsi route agar hanya dimuat sekali saat aplikasi dimulai
MODEL_PATH = "phishing_model.keras"
SCALER_PATH = "scaler.pkl" # Pastikan nama file dan path ini benar

model = None
scaler = None

try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"* Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"* ERROR: Model file not found at {MODEL_PATH}")

    if os.path.exists(SCALER_PATH):
        # Menggunakan joblib untuk memuat scaler.pkl (sesuai kode Anda sebelumnya)
        scaler = joblib.load(SCALER_PATH)
        print(f"* Scaler loaded successfully from {SCALER_PATH} (using joblib)")
        # Pastikan scaler yang dimuat adalah instance dari StandardScaler jika perlu type checking
        if not isinstance(scaler, StandardScaler):
             print(f"* WARNING: Loaded scaler from {SCALER_PATH} might not be a StandardScaler instance.")

        # # Jika Anda sebenarnya menggunakan pickle:
        # with open(SCALER_PATH, 'rb') as f:
        #     scaler = pickle.load(f)
        # print(f"* Scaler loaded successfully from {SCALER_PATH} (using pickle)")
    else:
        print(f"* ERROR: Scaler file not found at {SCALER_PATH}")

except Exception as e:
    print(f"* ERROR loading model or scaler: {e}")
    # Pertimbangkan untuk menghentikan aplikasi jika model/scaler gagal dimuat
    # raise SystemExit(f"Critical error loading resources: {e}")


# --- Fungsi Ekstraksi Fitur (Salin dari Notebook Anda) ---
def extract_features(url):
    """
    Ekstraksi fitur-fitur penting dari sebuah URL untuk deteksi phishing.
    """
    features = {}
    try:
        # Normalisasi URL sederhana
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        parsed_url = tldextract.extract(url)
        parsed_full = urlparse(url)

        # Fitur asli
        features['url_length'] = len(url)
        features['special_chars'] = len(re.findall(r'[!@#$%^&*(),.?\":{}|<>]', url))
        subdomain = parsed_url.subdomain
        features['subdomain_count'] = len(subdomain.split('.')) if subdomain else 0
        features['has_https'] = 1 if parsed_full.scheme == 'https' else 0 # Lebih baik cek scheme
        features['has_at_symbol'] = 1 if '@' in url else 0
        features['domain_length'] = len(parsed_url.domain)
        features['domain_has_number'] = 1 if any(char.isdigit() for char in parsed_url.domain) else 0
        digit_count = sum(1 for char in url if char.isdigit())
        features['digit_ratio'] = digit_count / len(url) if len(url) > 0 else 0
        suspicious_words = ['secure', 'account', 'banking', 'login', 'signin', 'verify', 'confirm', 'update', 'paypal', 'password']
        features['suspicious_words'] = sum(1 for word in suspicious_words if word.lower() in url.lower())
        # Menghitung titik hanya di bagian domain + subdomain (lebih akurat)
        hostname = parsed_full.hostname if parsed_full.hostname else ""
        features['dots_in_domain'] = hostname.count('.')

        # Fitur tambahan
        # Logika has_www dari notebook: 0 JIKA ADA 'www.', 1 JIKA TIDAK ADA 'www.'
        # Ini agak aneh, biasanya sebaliknya. Pastikan model dilatih sesuai ini.
        features['has_www'] = 0 if 'www.' in hostname else 1
        # Pengecekan IP Address yang lebih baik
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        features['is_ip_address'] = 1 if re.match(ip_pattern, hostname) else 0

        features['path_length'] = len(parsed_full.path)
        features['query_length'] = len(parsed_full.query)

        whitelist = ['google.com', 'github.com', 'microsoft.com'] # Hanya domain+suffix
        # Hindari memasukkan URL lengkap atau path di whitelist domain
        domain_full = f"{parsed_url.domain}.{parsed_url.suffix}" if parsed_url.domain and parsed_url.suffix else parsed_url.domain
        features['domain_in_whitelist'] = 1 if domain_full in whitelist else 0

    except Exception as e:
        print(f"Error extracting features for URL '{url}': {e}")
        # Return dictionary dengan nilai default jika error
        feature_keys = ['url_length', 'special_chars', 'subdomain_count', 'has_https',
                        'has_at_symbol', 'domain_length', 'domain_has_number',
                        'digit_ratio', 'suspicious_words', 'dots_in_domain', 'has_www',
                        'is_ip_address', 'path_length', 'query_length', 'domain_in_whitelist']
        features = {key: 0 for key in feature_keys} # Atau nilai lain yang sesuai

    # Pastikan jumlah fitur selalu sama (15 fitur sesuai urutan ini)
    feature_order = ['url_length', 'special_chars', 'subdomain_count', 'has_https',
                     'has_at_symbol', 'domain_length', 'domain_has_number',
                     'digit_ratio', 'suspicious_words', 'dots_in_domain', 'has_www',
                     'is_ip_address', 'path_length', 'query_length', 'domain_in_whitelist']
    # Mengurutkan dictionary berdasarkan feature_order untuk konsistensi
    ordered_features = {key: features.get(key, 0) for key in feature_order}

    return ordered_features # Kembalikan dictionary yang sudah terurut


# --- Endpoint API untuk Prediksi ---
@app.route('/predict', methods=['POST'])
def predict_phishing():
    # Periksa apakah model dan scaler berhasil dimuat
    if model is None or scaler is None:
        return jsonify({"error": "Model or Scaler not loaded properly on the server"}), 500

    # Dapatkan data JSON dari request
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "Payload JSON harus berisi key 'url'"}), 400
        url_to_predict = data['url'].strip()
    except Exception as e:
         return jsonify({"error": f"Invalid JSON payload: {e}"}), 400

    if not url_to_predict:
        return jsonify({"error": "'url' cannot be empty"}), 400

    try:
        # 1. Ekstraksi Fitur (mendapatkan dictionary terurut)
        features_dict = extract_features(url_to_predict)
        # Ambil nilainya sesuai urutan yang sudah dijamin oleh extract_features
        features_list = list(features_dict.values())
        features_array = np.array(features_list).reshape(1, -1)

        # Cek jumlah fitur sebelum scaling
        if features_array.shape[1] != scaler.n_features_in_:
             print(f"ERROR: Feature count mismatch. Expected {scaler.n_features_in_}, got {features_array.shape[1]}")
             return jsonify({"error": f"Feature extraction mismatch. Expected {scaler.n_features_in_} features."}), 500


        # 2. Scaling Fitur
        features_normalized = scaler.transform(features_array)

        # 3. Prediksi menggunakan model
        prediction_proba = model.predict(features_normalized)[0][0] # Ambil probabilitas kelas 1 (phishing)

        # 4. Interpretasi Hasil (sesuai logika notebook Anda)
        threshold = 0.5
        if prediction_proba > threshold:
            result = "PHISHING"
            confidence = float(prediction_proba) # Pastikan float untuk JSON
        else:
            result = "LEGITIMATE"
            confidence = float(1 - prediction_proba) # Pastikan float untuk JSON

        # 5. Siapkan Response JSON
        response = {
            "url": url_to_predict,
            "prediction": result,
            "confidence": round(confidence, 4), # Pembulatan confidence
            "probability_phishing": round(float(prediction_proba), 4), # Kirim juga probabilitas mentah
            "extracted_features": features_dict # Kirim fitur yang diekstrak (opsional, pastikan terurut)
        }
        return jsonify(response)

    except Exception as e:
        print(f"ERROR during prediction for URL '{url_to_predict}': {e}")
        # Jangan ekspos detail error internal ke user
        return jsonify({"error": "An internal error occurred during prediction."}), 500

if __name__ == '__main__':
    from os import environ
    port = int(environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)