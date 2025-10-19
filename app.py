import streamlit as st
from PIL import Image, ImageDraw
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import pickle
import numpy as np

# --- Import các thư viện HOG ---
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog

# ======================================================================
# MODEL 1: FASTER R-CNN (Phát hiện Lỗi)
# ======================================================================

# 1.1️⃣ Hàm load model (sửa num_classes = 4)
def load_model_fasterrcnn(weight_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    num_classes = 4  # 3 lớp + 1 background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# 1.2️⃣ Tải mô hình chỉ 1 lần
@st.cache_resource
def get_model_fasterrcnn():
    model_path = "fasterrcnn_phone_defect1910.pth"  # đường dẫn file .pth của bạn
    model = load_model_fasterrcnn(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("✅ Model 1 (Faster R-CNN) loaded and cached.")
    return model, device


# 1.3️⃣ Hàm dự đoán cho Web App (Chỉ trả về STATUS)
def predict_for_webapp(model, device, image_pil, score_thresh=0.6):
    transform = T.ToTensor()
    img_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]
        
    has_detection = False
    found_defect = False
    found_nonphone = False

    for score, label in zip(outputs["scores"], outputs["labels"]):
        if score > score_thresh:
            has_detection = True
            label_id = int(label.cpu().numpy())
            if label_id == 2:
                found_defect = True
            elif label_id == 3:
                found_nonphone = True

    # 1.4️⃣ Logic kết luận (Không trả về ảnh)
    if not has_detection or found_nonphone:
        return "NO_PHONE"
    elif found_defect:
        return "DEFECTIVE"
    else:
        return "NON_DEFECTIVE"


# ======================================================================
# MODEL 2: HOG + SOFTMAX (Phân loại)
# ======================================================================

# 2.1️⃣ --- CONFIG HOG (Phải giống hệt file train) ---
HOG_IMG_SIZE = (128, 64)
PIXELS_PER_CELL = (16, 16)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9
COLOR_BINS = 32
MODEL_PATH = os.path.join("outputs", "softmax_model_hog_hist.pkl")

# 2.2️⃣ --- Tải Model đã huấn luyện ---
@st.cache_resource
def load_model_hog():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Lỗi: Không tìm thấy file model HOG tại '{MODEL_PATH}'")
        return None  # Trả về None nếu lỗi
    print(f"Đang tải model HOG từ {MODEL_PATH}...")
    try:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        print("Tải model HOG thành công.")
        return model_data
    except Exception as e:
        st.error(f"Lỗi khi tải model HOG: {e}")
        return None # Trả về None nếu lỗi


# 2.3️⃣ --- Hàm tính Softmax (Lấy từ file train) ---
def softmax_np(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


# 2.4️⃣ --- Hàm trích xuất đặc trưng HOG (ĐÃ CẬP NHẬT) ---
def extract_hog_features(img_pil):
    try:
        img = np.array(img_pil)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # Loại bỏ kênh Alpha
        resized_img = resize(img, HOG_IMG_SIZE, anti_aliasing=True)
        gray_img = rgb2gray(resized_img) if resized_img.ndim == 3 else resized_img
        features_hog = hog(gray_img, orientations=ORIENTATIONS,
                           pixels_per_cell=PIXELS_PER_CELL,
                           cells_per_block=CELLS_PER_BLOCK,
                           block_norm='L2-Hys',
                           visualize=False,
                           transform_sqrt=True)
        if resized_img.ndim == 3 and resized_img.shape[2] == 3:
            img_uint8 = (resized_img * 255).astype(np.uint8)
            hist_r = np.histogram(img_uint8[:, :, 0], bins=COLOR_BINS, range=(0, 256))[0]
            hist_g = np.histogram(img_uint8[:, :, 1], bins=COLOR_BINS, range=(0, 256))[0]
            hist_b = np.histogram(img_uint8[:, :, 2], bins=COLOR_BINS, range=(0, 256))[0]
            features_color_raw = np.concatenate((hist_r, hist_g, hist_b))
            features_color = features_color_raw / (features_color_raw.sum() + 1e-6)
        else:
            features_color = np.zeros(COLOR_BINS * 3)
        features = np.concatenate((features_hog, features_color))
        return features
    except Exception as e:
        print(f"Lỗi khi trích xuất HOG: {e}")
        return None


# ======================================================================
# 5️⃣ Giao diện Streamlit Chính (Kết quả lên đầu)
# ======================================================================
st.set_page_config(layout="wide", page_title="Phone Analysis App")

st.title("📱 Ứng dụng Phân tích Điện thoại")
st.write("Tải lên một ảnh, cả hai mô hình sẽ cùng phân tích và hiển thị kết quả ngay bên dưới.")

# --- Tải cả hai model lên trước ---
model_rcnn, device_rcnn = get_model_fasterrcnn()
model_data_hog = load_model_hog()

# --- Tạo 1 file uploader duy nhất ---
uploaded_file = st.file_uploader("📤 Chọn một ảnh duy nhất", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mở ảnh MỘT LẦN
    image_pil = Image.open(uploaded_file).convert("RGB")

    # --- HIỂN THỊ KẾT QUẢ (Theo yêu cầu) ---
    st.header("🔍 Kết Quả Phân Tích")
    
    # Tạo 2 dòng trống để chứa kết quả
    # Chúng sẽ được lấp đầy sau khi model chạy xong
    result_placeholder_1 = st.empty()
    result_placeholder_2 = st.empty()

    # Thêm một đường kẻ
    st.divider() 

    # --- HIỂN THỊ ẢNH GỐC (Bên dưới kết quả) ---
    st.header("🖼️ Ảnh Gốc Đã Tải Lên")
    st.image(image_pil, caption="Ảnh gốc", use_container_width=True)
    
    # --- Xử lý Model 1 (Faster R-CNN) ---
    with st.spinner("Model 1 (Faster R-CNN) đang xử lý..."):
        detection_status = predict_for_webapp(model_rcnn, device_rcnn, image_pil.copy(), score_thresh=0.6)
        
        # Format kết quả Model 1 (ĐÃ SỬA TÊN)
        result_text_1 = "### 1. Model Faster R-CNN: "
        if detection_status == "DEFECTIVE":
            result_text_1 += "❌ **PHÁT HIỆN LỖI (VỠ/BẨN)**"
        elif detection_status == "NON_DEFECTIVE":
            result_text_1 += "✅ **KHÔNG LỖI**"
        elif detection_status == "NO_PHONE":
            result_text_1 += "⚠️ **KHÔNG PHÁT HIỆN ĐT**"
        
        # Đẩy kết quả vào placeholder 1
        result_placeholder_1.markdown(result_text_1)

    # --- Xử lý Model 2 (HOG + Softmax) ---
    with st.spinner("Model 2 (Softmax regression) đang xử lý..."):
        if model_data_hog is not None:
            # Giải nén các thành phần model HOG
            W = model_data_hog["W"]
            b = model_data_hog["b"]
            mean = model_data_hog["mean"]
            std = model_data_hog["std"]
            label_map = model_data_hog["label_map"]
            inv_label_map = {v: k for k, v in label_map.items()}

            features = extract_hog_features(image_pil.copy())
            
            # Format kết quả Model 2 (ĐÃ SỬA TÊN)
            result_text_2 = "### 2. Model Softmax regression: "
            if features is None:
                result_text_2 += "🚫 *Không thể xử lý ảnh này.*"
            else:
                features_2d = features.reshape(1, -1)
                if features_2d.shape[1] != mean.shape[1]:
                    result_text_2 += f"🚫 *Lỗi kích thước! (Cần {mean.shape[1]}, nhận được {features_2d.shape[1]})*"
                else:
                    # Chuẩn hóa và dự đoán
                    features_std = (features_2d - mean) / (std + 1e-12)
                    scores = features_std @ W + b
                    probs = softmax_np(scores)
                    pred_index = np.argmax(probs, axis=1)[0]
                    prediction_label = inv_label_map[pred_index]
                    probability = np.max(probs) * 100
                    
                    # Format kết quả Model 2
                    result_text_2 += f"**'{prediction_label}'** (Độ tin cậy: {probability:.2f}%)"
            
            # Đẩy kết quả vào placeholder 2
            result_placeholder_2.markdown(result_text_2)
        else:
            result_placeholder_2.error("### 2. Model Softmax regression: Lỗi tải model.")

else:
    # Thông báo chờ
    st.info("⬆️ Hãy tải một ảnh lên để cả hai mô hình cùng phân tích.")
