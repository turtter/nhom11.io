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

    # Load trọng số huấn luyện
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


# 1.3️⃣ Hàm dự đoán cho Web App
def predict_for_webapp(model, device, image_pil, score_thresh=0.6):
    transform = T.ToTensor()
    img_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    image_with_boxes = image_pil.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    # Map ID -> Nhãn
    label_map = {
        1: "KHÔNG LỖI",
        2: "BỊ LỖI",
        3: "KHÔNG PHẢI ĐIỆN THOẠI"
    }

    has_detection = False
    found_defect = False
    found_nonphone = False

    for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
        if score > score_thresh:
            has_detection = True
            box = box.cpu().numpy()
            label_id = int(label.cpu().numpy())

            # Màu khung
            if label_id == 1:
                color = "lime"
            elif label_id == 2:
                color = "red"
                found_defect = True
            elif label_id == 3:
                color = "blue"
                found_nonphone = True
            else:
                color = "white"

            # Vẽ khung và nhãn
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)
            text = f"{label_map.get(label_id, 'N/A')}: {score:.2f}"
            text_x, text_y = box[0], max(0, box[1] - 20)
            bbox = draw.textbbox((text_x, text_y), text)
            draw.rectangle(bbox, fill="black")
            draw.text((text_x, text_y), text, fill="yellow")

    # 1.4️⃣ Logic kết luận
    if not has_detection or found_nonphone:
        return "NO_PHONE", image_with_boxes
    elif found_defect:
        return "DEFECTIVE", image_with_boxes
    else:
        return "NON_DEFECTIVE", image_with_boxes


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
        st.error("Vui lòng đảm bảo file model đã được đẩy lên GitHub và nằm trong thư mục 'outputs'.")
        st.stop()  # Dừng ứng dụng

    print(f"Đang tải model HOG từ {MODEL_PATH}...")
    try:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        print("Tải model HOG thành công.")
        return model_data
    except FileNotFoundError:
        st.error(f"Lỗi FileNotFoundError: Không tìm thấy file model HOG tại '{MODEL_PATH}'.")
        st.error("Hãy kiểm tra lại đường dẫn và đảm bảo file đã được commit lên GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi tải model HOG: {e}")
        st.stop()


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

        # --- 1. Trích xuất HOG (từ ảnh xám) ---
        features_hog = hog(gray_img, orientations=ORIENTATIONS,
                           pixels_per_cell=PIXELS_PER_CELL,
                           cells_per_block=CELLS_PER_BLOCK,
                           block_norm='L2-Hys',
                           visualize=False,
                           transform_sqrt=True)

        # --- 2. Trích xuất Color Histogram (từ ảnh màu) ---
        if resized_img.ndim == 3 and resized_img.shape[2] == 3:
            img_uint8 = (resized_img * 255).astype(np.uint8)
            hist_r = np.histogram(img_uint8[:, :, 0], bins=COLOR_BINS, range=(0, 256))[0]
            hist_g = np.histogram(img_uint8[:, :, 1], bins=COLOR_BINS, range=(0, 256))[0]
            hist_b = np.histogram(img_uint8[:, :, 2], bins=COLOR_BINS, range=(0, 256))[0]
            features_color_raw = np.concatenate((hist_r, hist_g, hist_b))
            features_color = features_color_raw / (features_color_raw.sum() + 1e-6)
        else:
            features_color = np.zeros(COLOR_BINS * 3)

        # --- 3. Nối 2 đặc trưng ---
        features = np.concatenate((features_hog, features_color))
        return features

    except Exception as e:
        print(f"Lỗi khi trích xuất HOG: {e}")
        return None


# ======================================================================
# 5️⃣ Giao diện Streamlit Chính (Sử dụng Tabs)
# ======================================================================
st.set_page_config(layout="wide", page_title="Phone Analysis App")

st.title("📱 Ứng dụng Phân tích Điện thoại")
st.write("Sử dụng tab bên dưới để chọn mô hình bạn muốn dùng.")

tab1, tab2 = st.tabs([
    "Mô hình 1: Phát hiện Lỗi (Faster R-CNN)",
    "Mô hình 2: Phân loại (HOG + Histogram)"
])

# --- Giao diện cho Tab 1: Faster R-CNN ---
with tab1:
    st.header("1️⃣ Phát hiện Lỗi Điện thoại (3 lớp)")
    st.write("Tải lên ảnh điện thoại để mô hình phân loại: **KHÔNG LỖI**, **BỊ LỖI**, hoặc **KHÔNG PHÁT HIỆN RA ĐIỆN THOẠI**.")

    # Tải model 1
    model_rcnn, device_rcnn = get_model_fasterrcnn()

    col1, col2 = st.columns(2)

    with col1:
        # Thêm key="uploader1" để phân biệt với uploader ở tab 2
        uploaded_file_1 = st.file_uploader("📤 Chọn một ảnh (Model 1)", type=["jpg", "jpeg", "png"], key="uploader1")

    with col2:
        st.write("### 🔍 Kết quả dự đoán (Model 1)")

        if uploaded_file_1 is not None:
            image_1 = Image.open(uploaded_file_1).convert("RGB")
            with st.spinner("Đang xử lý (Model 1)..."):
                detection_status, result_image = predict_for_webapp(model_rcnn, device_rcnn, image_1, score_thresh=0.6)

                if detection_status == "DEFECTIVE":
                    st.error("❌ **KẾT QUẢ: PHÁT HIỆN ĐIỆN THOẠI BỊ LỖI (VỠ hoặc BẨN )**")
                elif detection_status == "NON_DEFECTIVE":
                    st.success("✅ **KẾT QUẢ: ĐIỆN THOẠI KHÔNG LỖI**")
                elif detection_status == "NO_PHONE":
                    st.warning("⚠️ **KẾT QUẢ: KHÔNG PHÁT HIỆN RA ĐIỆN THOẠI**")

                st.image(result_image, caption="Ảnh Kết Quả (Model 1)", use_container_width=True)
        else:
            st.info("⬆️ Hãy tải một ảnh lên cho Model 1 để xem kết quả.")


# --- Giao diện cho Tab 2: HOG + Softmax ---
with tab2:
    st.header("2️⃣ Phân loại ảnh điện thoại (HOG + Histogram)")
    
    # Tải model 2
    model_data_hog = load_model_hog()

    # Lấy các thành phần từ model
    W = model_data_hog["W"]
    b = model_data_hog["b"]
    mean = model_data_hog["mean"]
    std = model_data_hog["std"]
    label_map = model_data_hog["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}

    # 1. Tạo nút tải file (Thêm key="uploader2")
    uploaded_file_2 = st.file_uploader("Chọn một ảnh để dự đoán (Model 2):",
                                     type=["jpg", "jpeg", "png", "bmp"],
                                     key="uploader2")

    if uploaded_file_2 is not None:
        # 2. Hiển thị ảnh
        img_pil_2 = Image.open(uploaded_file_2)
        st.image(img_pil_2, caption="Ảnh đã tải lên", use_column_width=True)

        # 3. Tạo nút dự đoán
        if st.button("Dự đoán (Model 2)", key="button2"):
            # 4. Xử lý và dự đoán
            with st.spinner("Đang trích xuất đặc trưng và dự đoán (Model 2)..."):
                features = extract_hog_features(img_pil_2)

                if features is None:
                    st.error("Không thể xử lý ảnh này.")
                else:
                    # Chuẩn hóa đặc trưng
                    features_2d = features.reshape(1, -1)  # -> (1, 852)

                    # Kiểm tra kích thước trước khi chuẩn hóa
                    if features_2d.shape[1] != mean.shape[1]:
                        st.error(
                            f"Lỗi kích thước đặc trưng! Model mong đợi {mean.shape[1]}, nhưng nhận được {features_2d.shape[1]}.")
                    else:
                        features_std = (features_2d - mean) / (std + 1e-12)

                        # Dự đoán
                        scores = features_std @ W + b
                        probs = softmax_np(scores)

                        pred_index = np.argmax(probs, axis=1)[0]
                        prediction_label = inv_label_map[pred_index]
                        probability = np.max(probs) * 100

                        # 5. Hiển thị kết quả
                        st.success(f"**Kết quả (Model 2):** '{prediction_label}'")
                        st.info(f"**Độ tin cậy:** {probability:.2f}%")
