import cv2
import numpy as np
import easyocr
import re
import os
from ultralytics import YOLO

# --- CẤU HÌNH ---
YOLO_MODEL_PATH = "best.pt"
YOLO_CONF_THRESHOLD = 0.5

# Cấu hình cắt viền:
# Padding bên ngoài (để YOLO lấy rộng ra chút cho chắc chắn)
YOLO_PADDING = 0.05
# Cắt bỏ viền bên trong (để loại bỏ khung đen, ốc vít sau khi YOLO crop)
INNER_CROP_RATIO = 0.03  # Cắt bỏ 3% mỗi cạnh

# Biến toàn cục
YOLO_MODEL = None
READER = None


def load_models():
    """Hàm tải model, được gọi khi app khởi động"""
    global YOLO_MODEL, READER

    # 1. Load YOLO
    if os.path.exists(YOLO_MODEL_PATH):
        try:
            print(f"🔄 Đang tải YOLO từ {YOLO_MODEL_PATH}...")
            YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
            print("✅ Đã tải xong YOLO.")
        except Exception as e:
            print(f"❌ Lỗi tải YOLO: {e}")
            YOLO_MODEL = None
    else:
        print(f"⚠️ Không tìm thấy file {YOLO_MODEL_PATH}. Hãy chắc chắn bạn đã train model.")

    # 2. Load EasyOCR
    try:
        print("🔄 Đang khởi tạo EasyOCR...")
        # gpu=True nếu máy có card NVIDIA, nếu lỗi đổi thành False
        READER = easyocr.Reader(['en'], gpu=True)
        print("✅ Đã tải xong EasyOCR.")
    except Exception as e:
        print(f"❌ Lỗi tải EasyOCR: {e}")
        READER = None


def cleanup_text(text):
    """
    Lọc bỏ ký tự đặc biệt nhưng GIỮ LẠI dấu gạch ngang (-) và dấu chấm (.)
    để format sau này dễ hơn.
    """
    # Chỉ giữ A-Z, 0-9 và dấu - .
    text_clean = re.sub(r'[^A-Z0-9\-\.]', '', text.upper())
    return text_clean


def preprocess_plate(img_crop):
    """
    Xử lý ảnh biển số trước khi đưa vào OCR:
    1. Cắt bỏ viền (Inner Crop)
    2. Chuyển xám & Nhị phân hóa (Threshold)
    """
    if img_crop is None or img_crop.size == 0:
        return None

    h, w = img_crop.shape[:2]

    # 1. Cắt bỏ viền (Inner Crop) để loại bỏ khung xe/ốc vít
    # Tính số pixel cần cắt bỏ dựa trên tỷ lệ
    crop_h = int(h * INNER_CROP_RATIO)
    crop_w = int(w * INNER_CROP_RATIO)

    # Thực hiện cắt (nhớ kiểm tra để không cắt hết ảnh)
    if h > crop_h * 2 and w > crop_w * 2:
        img_crop = img_crop[crop_h:h - crop_h, crop_w:w - crop_w]

    # 2. Xử lý ảnh để làm rõ chữ
    # Chuyển xám
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

    # Tăng độ tương phản (Histogram Equalization) - Tốt cho ảnh tối
    # gray = cv2.equalizeHist(gray)

    # Nhị phân hóa (Thresholding): Chữ đen/trắng tách biệt
    # Dùng Otsu's thresholding để tự động tìm ngưỡng sáng tối ưu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def detect_plate_yolo(img):
    """Trả về (ảnh_crop_gốc, (x,y,w,h))"""
    if YOLO_MODEL is None:
        return None, None

    results = YOLO_MODEL(img, verbose=False)
    best_box = None
    best_conf = -1
    H, W = img.shape[:2]

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            if conf > best_conf and conf > YOLO_CONF_THRESHOLD:
                best_conf = conf
                best_box = box

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)

        # Mở rộng vùng chọn một chút (Padding) để không bị mất chữ ở mép
        w_box = x2 - x1
        h_box = y2 - y1
        pad_x = int(w_box * YOLO_PADDING)
        pad_y = int(h_box * YOLO_PADDING)

        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(W, x2 + pad_x)
        ny2 = min(H, y2 + pad_y)

        plate_crop = img[ny1:ny2, nx1:nx2].copy()
        return plate_crop, (nx1, ny1, nx2 - nx1, ny2 - ny1)

    return None, None


def format_plate_text(text):
    """
    Format lại biển số cho chuẩn VN.
    Ví dụ: 30A12345 -> 30A-123.45
    """
    # Xóa hết ký tự đặc biệt để format lại từ đầu cho chuẩn
    clean = re.sub(r'[^A-Z0-9]', '', text)

    if len(clean) < 4:
        return text  # Quá ngắn thì trả về nguyên gốc

    # Logic format biển dài (Biển 1 dòng): 29A12345 -> 29A-123.45
    # Logic format biển vuông (Biển 2 dòng): EasyOCR thường đọc nối liền -> 29A12345

    # Ví dụ đơn giản: Thêm gạch ngang sau ký tự thứ 3 (thường là mã tỉnh + seri)
    prefix = clean[:3]
    suffix = clean[3:]

    # Thêm dấu chấm vào phần số (nếu muốn đẹp: 12345 -> 123.45)
    if len(suffix) > 3:
        suffix = suffix[:-2] + '.' + suffix[-2:]

    return f"{prefix}-{suffix}"


def process_image_and_recognize(path_or_array):
    """
    Input: Đường dẫn ảnh hoặc mảng ảnh
    Output: (text_ket_qua, anh_bien_so_da_xu_ly, anh_toan_canh_ve_box)
    """
    if isinstance(path_or_array, str):
        img = cv2.imread(path_or_array)
    else:
        img = path_or_array

    if img is None:
        return "ERROR_LOAD", None, None

    img_display = img.copy()

    # 1. Detect YOLO
    plate_crop_raw, box = detect_plate_yolo(img)

    if plate_crop_raw is None:
        return "KHÔNG THẤY BIỂN", None, img_display

    # Vẽ box lên ảnh gốc
    x, y, w, h = box
    cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # 2. Xử lý ảnh (Cắt viền + Threshold)
    plate_processed = preprocess_plate(plate_crop_raw)

    # 3. EasyOCR Read
    if READER is None:
        return "LỖI MODEL OCR", plate_processed, img_display

    try:
        # allowlist: Giới hạn ký tự để OCR ít bị sai (chỉ đọc số và chữ in hoa, dấu gạch ngang)
        results = READER.readtext(
            plate_processed,
            detail=1,
            paragraph=False,
            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.'
        )

        # Sắp xếp kết quả:
        # Với biển vuông (2 dòng), OCR sẽ trả về 2 kết quả. Cần ghép lại.
        # Sắp xếp theo trục Y (trên xuống dưới), sau đó trục X (trái sang phải)
        results.sort(key=lambda r: r[0][0][1])

        full_text = ""
        for (bbox, text, prob) in results:
            if prob > 0.2:
                full_text += text

        # Làm sạch text (giữ lại dấu - nếu OCR đọc được)
        final_text = cleanup_text(full_text)

        # Nếu OCR không đọc được dấu -, ta dùng hàm format để tự điền
        if '-' not in final_text and len(final_text) > 6:
            final_text = format_plate_text(final_text)

        cv2.putText(img_display, final_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return final_text, plate_processed, img_display

    except Exception as e:
        print(f"Error OCR: {e}")
        return "LỖI XỬ LÝ", plate_processed, img_display


# --- MAIN TEST ---
if __name__ == "__main__":
    load_models()

    # Thay đường dẫn ảnh của bạn vào đây
    image_path = "test_xe.jpg"

    if os.path.exists(image_path):
        text, plate_img, result_img = process_image_and_recognize(image_path)
        print(f"Kết quả đọc: {text}")

        # Hiển thị
        cv2.imshow("Anh Goc + Box", result_img)
        if plate_img is not None:
            cv2.imshow("Bien So Da Xu Ly (Input cho OCR)", plate_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không tìm thấy file ảnh để test.")