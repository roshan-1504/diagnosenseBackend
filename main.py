from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import traceback
import time
import cv2
import xgboost as xgb
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, MobileNetV2, InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, concatenate

app = Flask(__name__)
CORS(app)

# ------------------ CKD Models ------------------
def convert_to_serializable(obj):
    """Convert NumPy data types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

# Load CKD models and preprocessing components
try:
    ckd_encoders = joblib.load("ckd_models/encoders.pkl")
    ckd_imputer = joblib.load("ckd_models/imputer.pkl")
    ckd_scaler = joblib.load("ckd_models/scaler.pkl")

    ckd_model_files = {
        'Logistic Regression': 'ckd_models/logistic_regression_model.pkl',
        'Random Forest': 'ckd_models/random_forest_model.pkl',
        'XGBoost': 'ckd_models/xgboost_model.pkl'
    }
    ckd_models = {name: joblib.load(path) for name, path in ckd_model_files.items()}
    ckd_categorical_columns = ['htn', 'dm', 'pc']
    ckd_selected_features = ['sg', 'al', 'sc', 'bu', 'hemo', 'htn', 'dm', 'pc', 'bgr']
    print("CKD models loaded successfully")
except Exception as e:
    print(f"Error loading CKD models: {e}")
    ckd_models_loaded = False
else:
    ckd_models_loaded = True

# ------------------ Pneumonia Detection Models ------------------
try:
    pd_rf_model = joblib.load('pneumonia_models/pd_rf_model.joblib')
    pd_knn_model = joblib.load('pneumonia_models/pd_knn_model.joblib')
    pd_xgb_model = joblib.load('pneumonia_models/pd_xgboost_model.joblib')

    input_shape = (224, 224, 3)

    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    inception_base = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    resnet_features = GlobalAveragePooling2D()(resnet_base.output)
    mobilenet_features = GlobalAveragePooling2D()(mobilenet_base.output)
    inception_features = GlobalAveragePooling2D()(inception_base.output)

    combined_features = concatenate([resnet_features, mobilenet_features, inception_features])

    feature_extractor = Model(
        inputs=[resnet_base.input, mobilenet_base.input, inception_base.input],
        outputs=combined_features
    )
    print("Pneumonia detection models loaded successfully")
except Exception as e:
    print(f"Error loading pneumonia detection models: {e}")
    pd_models_loaded = False
else:
    pd_models_loaded = True

# ------------------ Prostate Cancer Models ------------------
try:
    pc_scaler = joblib.load("prostate_cancer_models/scaler.pkl")
    pc_top_indices = joblib.load("prostate_cancer_models/selected_genes.pkl")

    pc_models = {
        "Logistic Regression": joblib.load("prostate_cancer_models/pc_lr_model.pkl"),
        "XGBoost": joblib.load("prostate_cancer_models/pc_xgb_model.pkl"),
        "Random Forest": joblib.load("prostate_cancer_models/pc_rf_model.pkl")
    }
    print("Prostate cancer models loaded successfully")
except Exception as e:
    print(f"Error loading prostate cancer models: {e}")
    pc_models_loaded = False
else:
    pc_models_loaded = True

# ------------------ Pneumonia Detection Helper Functions ------------------
def preprocess_image(image):
    """
    Resize and normalize image to feed into CNNs.
    """
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = image.astype("float32") / 255.0
    return image

def extract_features_from_image(image):
    """
    Extract deep features from image using the combined CNN feature extractor.
    """
    image_batch = np.expand_dims(image, axis=0)
    features = feature_extractor.predict([image_batch] * 3)  # same image for all 3 CNNs
    return features.reshape(1, -1)

# ------------------ CKD ENDPOINT ------------------
@app.route("/predict_ckd", methods=["POST"])
def predict_ckd():
    if not ckd_models_loaded:
        return jsonify({"status": "error", "message": "CKD models not loaded properly"}), 500
    
    try:
        data = request.get_json()
        print("Incoming JSON data:", data)

        missing = [feat for feat in ckd_selected_features if feat not in data]
        if missing:
            return jsonify({
                "status": "error",
                "message": f"Missing required features: {', '.join(missing)}",
                "received_data": data
            }), 400

        input_data = {feat: data.get(feat) for feat in ckd_selected_features}
        df = pd.DataFrame([input_data])
        print("Input DataFrame:\n", df)

        for col in ckd_categorical_columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            try:
                df[col] = ckd_encoders[col].transform(df[col])
            except Exception as e:
                return jsonify({
                    "status": "error",
                    "message": f"Invalid value for '{col}': {str(e)}",
                    "valid_values": list(ckd_encoders[col].classes_)
                }), 400

        print("Encoded DataFrame:\n", df)

        try:
            df_imputed = pd.DataFrame(ckd_imputer.transform(df), columns=df.columns)
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Imputation failed: {str(e)}"
            }), 400

        try:
            df_scaled = ckd_scaler.transform(df_imputed)
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Scaling failed: {str(e)}"
            }), 400

        print("Scaled data:", df_scaled)

        results = {}
        votes = []  

        for name, model in ckd_models.items():
            try:
                prediction = model.predict(df_scaled)[0]
                probability = model.predict_proba(df_scaled)[0][1]
                label = "There is a high risk of Chronic Kidney Disease" if prediction == 1 else "There is no risk of Chronic Kidney Disease"
                votes.append(label)

                results[name] = {
                    "prediction": label,
                    "probability_ckd": round(float(probability), 4)
                }
            except Exception as e:
                results[name] = {
                    "error": f"Model error: {str(e)}"
                }

        from collections import Counter
        vote_counts = Counter(votes)
        majority_vote = vote_counts.most_common(1)[0][0]
        results["Majority Vote Result"] = majority_vote

        return jsonify({
            "status": "success",
            "input": convert_to_serializable(data),
            "predictions": convert_to_serializable(results)
        })

    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Unhandled exception:\n", traceback_str)
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback_str
        }), 500

# ------------------ PNEUMONIA DETECTION ENDPOINT ------------------
@app.route('/predict_pneumonia', methods=['POST'])
def predict_pneumonia():
    if not pd_models_loaded:
        return jsonify({"error": "Pneumonia detection models not loaded properly"}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    start_time = time.time()

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    image = preprocess_image(image)
    features = extract_features_from_image(image)

    rf_proba = pd_rf_model.predict_proba(features)[0]
    knn_proba = pd_knn_model.predict_proba(features)[0]
    xgb_proba = float(pd_xgb_model.predict(xgb.DMatrix(features)))

    rf_pred = int(rf_proba[1] > 0.5)
    knn_pred = int(knn_proba[1] > 0.5)
    xgb_pred = int(xgb_proba > 0.5)

    predictions = [rf_pred, knn_pred, xgb_pred]
    majority_vote = 1 if predictions.count(1) >= 2 else 0

    label_map = {0: "NORMAL", 1: "PNEUMONIA"}

    end_time = time.time()
    inference_time = round(end_time - start_time, 2)

    return jsonify({
        'Random Forest': {
            "PNEUMONIA": round(rf_proba[1], 4),
            "NORMAL": round(rf_proba[0], 4),
            "Prediction": label_map[rf_pred]
        },
        'KNN': {
            "PNEUMONIA": round(knn_proba[1], 4),
            "NORMAL": round(knn_proba[0], 4),
            "Prediction": label_map[knn_pred]
        },
        'XGBoost': {
            "PNEUMONIA": round(xgb_proba, 4),
            "NORMAL": round(1 - xgb_proba, 4),
            "Prediction": label_map[xgb_pred]
        },
        'Majority Vote Result': label_map[majority_vote],
        'Inference Time (s)': inference_time
    })

# ------------------ PROSTATE CANCER ENDPOINT ------------------
@app.route("/predict_prostate_cancer", methods=["POST"])
def predict_prostate_cancer():
    if not pc_models_loaded:
        return jsonify({"error": "Prostate cancer models not loaded properly"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Load CSV as DataFrame
        new_patient = pd.read_csv(file)

        # Preprocess: scale and select features
        X_scaled = pc_scaler.transform(new_patient)
        X_selected = X_scaled[:, pc_top_indices]

        results = {}
        votes = []

        for model_name, model in pc_models.items():
            pred = model.predict(X_selected)[0]
            proba = model.predict_proba(X_selected)[0]

            cancer_prob = float(proba[1])
            non_cancer_prob = float(proba[0])
            label = "Cancer" if pred == 1 else "Non-Cancer"
            votes.append(label)

            results[model_name] = {
                "Prediction": label,
                "Probabilities": {
                    "Cancer": cancer_prob,
                    "Non-Cancer": non_cancer_prob
                }
            }

        # Majority voting for final prediction
        final_prediction = max(set(votes), key=votes.count)
        results["Majority Vote Result"] = final_prediction

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# # Health check endpoint
# @app.route("/health", methods=["GET"])
# def health_check():
#     status = {
#         "status": "healthy",
#         "models_loaded": {
#             "ckd_models": ckd_models_loaded,
#             "pneumonia_detection_models": pd_models_loaded,
#             "prostate_cancer_models": pc_models_loaded
#         }
#     }
#     return jsonify(status)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
