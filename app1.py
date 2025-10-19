import streamlit as st
from PIL import Image, ImageDraw
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ================================
# 1️⃣ Hàm load model (sửa num_classes = 4)
# ================================
def load_model(weight_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )

    # 3 lớp (non_defective_phone, defective, non-phone) + 1 background
    num_classes = 4

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load trọng số huấn luyện
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model


# ================================
# 2️⃣ Tải mô hình chỉ 1 lần
# ================================
@st.cache_resource
def get_model():
    model_path = "fasterrcnn_phone_defect1910.pth"  # đường dẫn file .pth của bạn
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("✅ Model loaded and cached.")
    return model, device


# ================================
# 3️⃣ Hàm dự đoán cho Web App
# ================================
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

    # ================================
    # 4️⃣ Logic kết luận
    # ================================
    if not has_detection or found_nonphone:
        return "NO_PHONE", image_with_boxes
    elif found_defect:
        return "DEFECTIVE", image_with_boxes
    else:
        return "NON_DEFECTIVE", image_with_boxes


# ================================
# 5️⃣ Giao diện Streamlit
# ================================
st.set_page_config(layout="wide", page_title="Phone Defect Detection")

st.title("📱 Ứng dụng Phát hiện Lỗi Điện thoại (3 lớp)")
st.write("Tải lên ảnh điện thoại để mô hình phân loại: **KHÔNG LỖI**, **BỊ LỖI**, hoặc **KHÔNG PHÁT HIỆN RA ĐIỆN THOẠI**.")

model, device = get_model()

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📤 Chọn một ảnh", type=["jpg", "jpeg", "png"])

with col2:
    st.write("### 🔍 Kết quả dự đoán")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        with st.spinner("Đang xử lý..."):
            detection_status, result_image = predict_for_webapp(model, device, image, score_thresh=0.6)

            if detection_status == "DEFECTIVE":
                st.error("❌ **KẾT QUẢ: PHÁT HIỆN ĐIỆN THOẠI BỊ LỖI (VỠ hoặc BẨN )**")
            elif detection_status == "NON_DEFECTIVE":
                st.success("✅ **KẾT QUẢ: ĐIỆN THOẠI KHÔNG LỖI**")
            elif detection_status == "NO_PHONE":
                st.warning("⚠️ **KẾT QUẢ: KHÔNG PHÁT HIỆN RA ĐIỆN THOẠI**")

            st.image(result_image, caption="Ảnh Kết Quả", use_container_width=True)
    else:
        st.info("⬆️ Hãy tải một ảnh lên để xem kết quả.")
