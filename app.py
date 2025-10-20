# -*- coding: utf-8 -*-
import streamlit as st
import torch
import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.svm import SVC
from PIL import Image
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pickle
import torch.nn as nn
from torchvision import models
import joblib

# ======================================================================
# GIẢI PHÁP MỚI: TÌM ĐƯỜNG DẪN TUYỆT ĐỐI ĐẾN THƯ MỤC CHỨA SCRIPT
# ======================================================================
# Lấy đường dẫn đến thư mục chứa file app.py này
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================================
# MODEL 1: FASTER R-CNN
# ======================================================================
@st.cache_resource
def get_model_fasterrcnn():
    # Sử dụng đường dẫn tuyệt đối
    model_path = os.path.join(BASE_DIR, "fasterrcnn_phone_defect1910.pth")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    num_classes = 4
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    try:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.error(f"Lỗi Model 1: Không tìm thấy file '{model_path}'.")
        return None, None
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("✅ Model 1 (Faster R-CNN) loaded.")
    return model, device

def predict_fasterrcnn(model, device, image_pil, score_thresh=0.6):
    transform = T.ToTensor()
    img_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)[0]
    has_detection, found_defect, found_nonphone = False, False, False
    for score, label in zip(outputs["scores"], outputs["labels"]):
        if score > score_thresh:
            has_detection = True
            label_id = int(label.cpu().numpy())
            if label_id == 2: found_defect = True
            elif label_id == 3: found_nonphone = True
    if not has_detection or found_nonphone: return "NO_PHONE"
    elif found_defect: return "DEFECTIVE"
    else: return "NON_DEFECTIVE"

# ======================================================================
# MODEL 2: SOFTMAX REGRESSION
# ======================================================================
@st.cache_resource
def load_model_softmax():
    # Sử dụng đường dẫn tuyệt đối
    model_path = os.path.join(BASE_DIR, "outputs", "softmax_model_hog_hist.pkl")
    if not os.path.exists(model_path):
        st.error(f"Lỗi Model 2: Không tìm thấy file '{model_path}'.")
        return None
    try:
        with open(model_path, "rb") as f: model_data = pickle.load(f)
        print("✅ Model 2 (Softmax) loaded.")
        return model_data
    except Exception as e:
        st.error(f"Lỗi khi tải Model 2: {e}")
        return None

def softmax_np(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def extract_features_softmax(img_pil):
    try:
        from skimage.transform import resize
        from skimage.color import rgb2gray
        img = np.array(img_pil)
        if img.ndim == 3 and img.shape[2] == 4: img = img[:, :, :3]
        resized_img = resize(img, (128, 64), anti_aliasing=True)
        gray_img = rgb2gray(resized_img) if resized_img.ndim == 3 else resized_img
        features_hog = hog(gray_img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False, transform_sqrt=True)
        if resized_img.ndim == 3 and resized_img.shape[2] == 3:
            img_uint8 = (resized_img * 255).astype(np.uint8)
            hist_r, hist_g, hist_b = (np.histogram(img_uint8[:, :, i], bins=32, range=(0, 256))[0] for i in range(3))
            features_color_raw = np.concatenate((hist_r, hist_g, hist_b))
            features_color = features_color_raw / (features_color_raw.sum() + 1e-6)
        else:
            features_color = np.zeros(32 * 3)
        return np.concatenate((features_hog, features_color))
    except Exception as e:
        print(f"Lỗi trích xuất đặc trưng Model 2: {e}")
        return None

# ======================================================================
# MODEL 3: SVM
# ======================================================================
@st.cache_resource
def load_model_svm():
    # Sử dụng đường dẫn tuyệt đối
    model_path = os.path.join(BASE_DIR, "svm_model.pth")
    if not os.path.exists(model_path):
        st.error(f"Lỗi Model 3: Không tìm thấy file '{model_path}'.")
        return None
    try:
        model = torch.load(model_path, weights_only=False)
        print("✅ Model 3 (SVM) loaded.")
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải Model 3: {e}")
        return None

def extract_features_svm(img_pil):
    try:
        img_array = np.array(img_pil)
        if img_array.shape[2] == 4: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img_gray, (128, 128))
        features = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
        std_dev = np.std(img)
        return np.hstack((features, std_dev))
    except Exception as e:
        print(f"Lỗi trích xuất đặc trưng Model 3: {e}")
        return None

# ======================================================================
# MODEL 4: RANDOM FOREST
# ======================================================================
@st.cache_resource
def load_model_rf():
    # Sử dụng đường dẫn tuyệt đối
    rf_path = os.path.join(BASE_DIR, "rf_clean.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler_clean.pkl")
    
    if not os.path.exists(rf_path):
        st.error(f"Lỗi Model 4: Không tìm thấy file '{rf_path}'.")
        return None, None

    if not os.path.exists(scaler_path):
        st.error(f"Lỗi Model 4: Không tìm thấy file '{scaler_path}'.")
        return None, None

    try:
        rf = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)
        print("✅ Model 4 (Random Forest) loaded.")
        return rf, scaler
    except Exception as e:
        st.error(f"Lỗi khi tải Model 4: {e}")
        return None, None
        
@st.cache_resource
def get_feature_extractor_rf():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return feature_extractor, transform, device

def predict_rf(rf_model, scaler, feature_extractor, transform, device, image_pil):
    try:
        img_tensor = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = feature_extractor(img_tensor)
        vec = feat.cpu().numpy().flatten()
        vec_scaled = scaler.transform([vec])
        pred = rf_model.predict(vec_scaled)[0]
        prob = rf_model.predict_proba(vec_scaled)[0]
        
        label_map = {0: "Non-Defective", 1: "Defective", 2: "Non-Phone"}
        pred_int = int(pred)
        label = label_map.get(pred_int, "Unknown")
        confidence = prob[pred_int] * 100 if pred_int < len(prob) else np.max(prob) * 100
        return label, confidence
    except Exception as e:
        print(f"Lỗi dự đoán Model 4: {e}")
        return None, None

# ======================================================================
# GIAO DIỆN STREAMLIT
# ======================================================================
st.set_page_config(layout="wide", page_title="Phone Analysis App")
st.title("📱 Ứng dụng Phân tích Điện thoại (4 Model)")
st.write("Tải lên một ảnh, cả bốn mô hình sẽ cùng phân tích và hiển thị kết quả.")

# --- Tải cả bốn model ---
model_rcnn, device_rcnn = get_model_fasterrcnn()
model_data_softmax = load_model_softmax()
model_svm = load_model_svm()
model_rf, scaler_rf = load_model_rf()
feature_extractor_rf, transform_rf, device_rf = get_feature_extractor_rf()

# --- Giao diện tải file ---
uploaded_file = st.file_uploader("📤 Chọn một ảnh duy nhất", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")

    st.header("🔍 Kết Quả Phân Tích")
    result_placeholder_1 = st.empty()
    result_placeholder_2 = st.empty()
    result_placeholder_3 = st.empty()
    result_placeholder_4 = st.empty()
    st.divider()

    st.header("🖼️ Ảnh Gốc Đã Tải Lên")
    st.image(image_pil, caption="Ảnh gốc", use_container_width=True)

    # --- Xử lý Model 1: Faster R-CNN ---
    if model_rcnn:
        with st.spinner("Model 1 (Faster R-CNN) đang xử lý..."):
            status = predict_fasterrcnn(model_rcnn, device_rcnn, image_pil.copy())
            text = "### 1. Model Faster R-CNN: "
            if status == "DEFECTIVE": text += "❌ **PHÁT HIỆN LỖI (VỠ/BẨN)**"
            elif status == "NON_DEFECTIVE": text += "✅ **KHÔNG LỖI**"
            else: text += "⚠️ **KHÔNG PHÁT HIỆN ĐT**"
            result_placeholder_1.markdown(text)

    # --- Xử lý Model 2: Softmax Regression ---
    if model_data_softmax:
        with st.spinner("Model 2 (Softmax regression) đang xử lý..."):
            features = extract_features_softmax(image_pil.copy())
            text = "### 2. Model Softmax regression: "
            if features is None: text += "🚫 *Không thể xử lý ảnh này.*"
            else:
                W, b, mean, std = model_data_softmax["W"], model_data_softmax["b"], model_data_softmax["mean"], model_data_softmax["std"]
                inv_label_map = {v: k for k, v in model_data_softmax["label_map"].items()}
                features_2d = features.reshape(1, -1)
                if features_2d.shape[1] != mean.shape[1]: text += f"🚫 *Lỗi kích thước!*"
                else:
                    features_std = (features_2d - mean) / (std + 1e-12)
                    scores = features_std @ W + b
                    probs = softmax_np(scores)
                    pred_index = np.argmax(probs, axis=1)[0]
                    prediction_label = inv_label_map[pred_index]
                    probability = np.max(probs) * 100
                    text += f"**'{prediction_label}'** (Độ tin cậy: {probability:.2f}%)"
            result_placeholder_2.markdown(text)

    # --- Xử lý Model 3: SVM ---
    if model_svm:
        with st.spinner("Model 3 (SVM) đang xử lý..."):
            features = extract_features_svm(image_pil.copy())
            text = "### 3. Model SVM: "
            if features is None: text += "🚫 *Không thể xử lý ảnh này.*"
            else:
                CLASS_NAMES = {0: 'Không phải điện thoại', 1: 'Không lỗi', 2: 'Bị lỗi/Vỡ'}
                features_2d = features.reshape(1, -1)
                prediction_id = model_svm.predict(features_2d)[0]
                prediction_name = CLASS_NAMES.get(prediction_id, "Không xác định")
                if prediction_id == 2: icon = "❌"
                elif prediction_id == 1: icon = "✅"
                else: icon = "⚠️"
                text += f"{icon} **{prediction_name}**"
            result_placeholder_3.markdown(text)
            
    # --- Xử lý Model 4: Random Forest ---
    if model_rf and scaler_rf:
        with st.spinner("Model 4 (Random Forest) đang xử lý..."):
            label, confidence = predict_rf(model_rf, scaler_rf, feature_extractor_rf, transform_rf, device_rf, image_pil.copy())
            text = "### 4. Model Random Forest: "
            if label is None:
                text += "🚫 *Lỗi trong quá trình dự đoán.*"
            else:
                label_map_display = {
                    "Defective": ("Bị lỗi/Vỡ", "❌"),
                    "Non-Defective": ("Không lỗi", "✅"),
                    "Non-Phone": ("Không phải điện thoại", "⚠️"),
                }
                display_name, icon = label_map_display.get(label, (label, "❓"))
                text += f"{icon} **{display_name}** (Độ tin cậy: {confidence:.2f}%)"
            result_placeholder_4.markdown(text)
else:
    st.info("⬆️ Hãy tải một ảnh lên để cả bốn mô hình cùng phân tích.")

