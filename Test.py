import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import json
import os
import time

# ===================== TRY IMPORT EASYOCR =====================
try:
    import easyocr

    EASYOCR_AVAILABLE = True
    print("✅ EasyOCR available - will be used as backup")
    # Khởi tạo EasyOCR reader (Vietnamese và English)
    try:
        # Using 'en' only since license plates are typically alphanumeric (English characters)
        # Setting gpu=False for broader compatibility unless a GPU is explicitly available and configured
        ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("✅ EasyOCR Reader initialized")
    except Exception as e:
        print(f"⚠️ EasyOCR init failed: {e}")
        EASYOCR_AVAILABLE = False
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️ EasyOCR not installed. Install with: pip install easyocr")
    print("   Running with CNN only...")

# ===================== Load model =====================
model_loaded = False
# Prioritize models known to be better if you have a specific order
model_paths = ["best_cnn_model.h5", "best_ann.h5", "BestANNFinals.h5", "best_CNN_model1.h5","CNN_final_trained.h5"]
loaded_model_path = ""
model = None # Initialize model to None
for model_path in model_paths:
    if os.path.exists(model_path):
        try:
            # Ensure custom objects are handled if your model uses them (e.g., custom layers, activations)
            # Example: custom_objects={"swish": tf.keras.activations.swish}
            model = tf.keras.models.load_model(model_path)
            print(f"✅ Loaded model: {model_path}")
            print(f"   Input shape: {model.input_shape}")
            model_loaded = True
            loaded_model_path = model_path
            break
        except Exception as e:
            print(f"❌ Error loading {model_path}: {e}")
            continue

if not model_loaded:
    messagebox.showerror("Error", "No model found. Please train first!")
    exit()

# ===================== Characters =====================
characters = list("0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ") # Default
try:
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    # Reconstruct characters list based on sorted indices to ensure correct mapping
    index_to_char = {int(v): k for k, v in class_indices.items()} # Ensure keys are int
    characters = [index_to_char[i] for i in sorted(index_to_char.keys())]
    print(f"✅ Loaded {len(characters)} classes from JSON: {characters}")
except FileNotFoundError:
    print("⚠️ class_indices.json not found. Using default characters.")
except Exception as e:
    print(f"⚠️ Error reading JSON: {e}. Using default characters.")

# ===================== Province codes =====================
province_dict = {
    "11": "Cao Bằng", "12": "Lạng Sơn", "14": "Quảng Ninh",
    "15": "Hải Phòng", "16": "Hải Phòng", "17": "Thái Bình",
    "18": "Nam Định", "19": "Phú Thọ", "20": "Thái Nguyên",
    "21": "Yên Bái", "22": "Tuyên Quang", "23": "Hà Giang",
    "24": "Lào Cai", "25": "Lai Châu", "26": "Sơn La",
    "27": "Điện Biên", "28": "Hòa Bình", "29": "Hà Nội",
    "30": "Hà Nội", "31": "Hà Nội", "32": "Hà Nội",
    "33": "Hà Tây (cũ)", "34": "Hải Dương", "35": "Ninh Bình",
    "36": "Thanh Hóa", "37": "Nghệ An", "38": "Hà Tĩnh",
    "43": "Đà Nẵng", "47": "Đắk Lắk", "48": "Đắk Nông",
    "49": "Lâm Đồng", "50": "TP.HCM", "51": "TP.HCM",
    "52": "TP.HCM", "53": "TP.HCM", "54": "TP.HCM",
    "55": "TP.HCM", "56": "TP.HCM", "57": "TP.HCM",
    "58": "TP.HCM", "59": "TP.HCM", "60": "Đồng Nai",
    "61": "Bình Dương", "62": "Long An", "63": "Tiền Giang",
    "64": "Vĩnh Long", "65": "Cần Thơ", "66": "Đồng Tháp",
    "67": "An Giang", "68": "Kiên Giang", "69": "Cà Mau",
    "70": "Tây Ninh", "71": "Bến Tre", "72": "Bà Rịa-Vũng Tàu",
    "73": "Quảng Bình", "74": "Quảng Trị", "75": "Thừa Thiên Huế",
    "76": "Quảng Ngãi", "77": "Bình Định", "78": "Phú Yên",
    "79": "Khánh Hòa", "81": "Gia Lai", "82": "Kon Tum",
    "83": "Sóc Trăng", "84": "Trà Vinh", "85": "Ninh Thuận",
    "86": "Bình Thuận", "88": "Vĩnh Phúc", "89": "Hưng Yên",
    "90": "Hà Nam", "92": "Quảng Nam", "93": "Bình Phước",
    "94": "Bạc Liêu", "95": "Hậu Giang", "97": "Bắc Kạn",
    "98": "Bắc Giang", "99": "Bắc Ninh"
}


# ===================== Preprocessing =====================
def remove_red_border(plate_img):
    if plate_img is None or plate_img.size == 0:
        return plate_img

    # Convert to HSV to better isolate red color
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    # Define range for red color in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)
    red_mask = cv2.erode(red_mask, kernel, iterations=1)

    if cv2.countNonZero(red_mask) > (
            plate_img.shape[0] * plate_img.shape[1] * 0.005):  # Only remove if significant red area
        mask_inv = cv2.bitwise_not(red_mask)
        # Apply mask to keep non-red regions, making red regions black
        plate_img = cv2.bitwise_and(plate_img, plate_img, mask=mask_inv)
        print("  🔴 Removed red border")

    return plate_img


def preprocess_plate_binary(plate_img):
    if plate_img is None or plate_img.size == 0:
        return None

    plate_img = remove_red_border(plate_img)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Tăng cường độ tương phản cục bộ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)

    # Sử dụng ngưỡng thích ứng, rất hiệu quả với điều kiện ánh sáng không đồng đều
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 19, 9)

    # Tinh chỉnh các thao tác hình thái học để nối các ký tự bị đứt và loại bỏ nhiễu
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return binary


def preprocess_char(img, img_size=(28, 28)):
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        return None

    # 1. Chuyển sang ảnh xám
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Ngưỡng hóa để có ảnh nhị phân nền đen (0), chữ trắng (255)
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Tìm contour lớn nhất (chính là ký tự) để cắt bỏ khoảng trắng thừa
    try:
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            char_crop = binary_img[y:y + h, x:x + w]
        else:
            char_crop = binary_img
    except Exception:
        char_crop = binary_img

    # 4. Tạo một canvas vuông ĐEN và căn giữa ký tự vào đó
    h, w = char_crop.shape
    new_size = max(h, w) + 4  # Thêm 4 pixel đệm

    # Tạo một canvas vuông ĐEN (giá trị 0)
    square_canvas = np.zeros((new_size, new_size), np.uint8)

    # Tính toán vị trí để paste ký tự vào giữa canvas
    paste_x = (new_size - w) // 2
    paste_y = (new_size - h) // 2

    square_canvas[paste_y:paste_y + h, paste_x:paste_x + w] = char_crop

    # 5. Resize về kích thước chuẩn của mô hình
    resized_img = cv2.resize(square_canvas, img_size, interpolation=cv2.INTER_AREA)

    # 6. Chuẩn hóa và reshape cho mô hình
    final_img = resized_img.astype("float32") / 255.0
    final_img = final_img.reshape(1, img_size[0], img_size[1], 1)

    return final_img
def predict_char_cnn(img):
    processed_img = preprocess_char(img)
    if processed_img is None:
        return "?", 0.0

    pred = model.predict(processed_img, verbose=0)
    confidence = np.max(pred)
    index = np.argmax(pred)

    # Lower confidence threshold for initial CNN prediction, let hybrid decide
    if confidence > 0.1:  # Even low confidence might be a hint
        return characters[index], confidence
    return "?", confidence


ocr_reader = None

def predict_char_easyocr(img):
    global ocr_reader # Khai báo sử dụng biến toàn cục
    if not EASYOCR_AVAILABLE:
        return "?", 0.0

    # Chỉ khởi tạo một lần duy nhất
    if ocr_reader is None:
        print("Initializing EasyOCR Reader for the first time...")
        ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("EasyOCR Reader initialized.")

    try:
        # EasyOCR needs larger, color images for best performance
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_rgb = img

        # Resize for better OCR quality. Target height of ~60-100 pixels is good.
        h, w = img_rgb.shape[:2]
        scale_factor = 80.0 / h  # Target height of 80 pixels
        img_resized = cv2.resize(img_rgb, (int(w * scale_factor), 80), interpolation=cv2.INTER_CUBIC)

        # Allow only alphanumeric characters relevant for license plates
        # EasyOCR's readtext returns (bbox, text, confidence)
        results = ocr_reader.readtext(img_resized, detail=1, allowlist='0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ',
                                      paragraph=False)

        if results and len(results) > 0:
            # Sort by confidence descending and take the best result
            results.sort(key=lambda x: x[2], reverse=True)
            best_text = results[0][1].strip().upper()
            best_conf = results[0][2]

            if len(best_text) > 0:
                # If EasyOCR returns multiple characters (e.g., 'AB' for 'A'),
                # usually the first one is the intended single character.
                return best_text[0], best_conf

        return "?", 0.0
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return "?", 0.0


def post_process_plate_text(text):
    """
    Sửa lỗi dựa trên cấu trúc biển số xe Việt Nam (VD: 30G-53507)
    """
    if not text or len(text) < 7:
        return text

    char_list = list(text)

    # === Xử lý 3 ký tự đầu (VD: 30G) ===
    # Vị trí 0 (số): 'G'->'6', 'B'->'8'
    if char_list[0] == 'G': char_list[0] = '6'
    if char_list[0] == 'B': char_list[0] = '8'

    # Vị trí 1 (số): 'D'->'0', 'G'->'0', 'B'->'8'
    if char_list[1] == 'D': char_list[1] = '0'
    if char_list[1] == 'G': char_list[1] = '0'
    if char_list[1] == 'B': char_list[1] = '8'

    # Vị trí 2 (chữ): '0'->'D', '8'->'B', '6'->'G'
    if char_list[2] == '0': char_list[2] = 'D'
    if char_list[2] == '8': char_list[2] = 'B'
    if char_list[2] == '6': char_list[2] = 'G'

    # === Xử lý các ký tự cuối (phải là số) ===
    for i in range(3, len(char_list)):
        # Sửa các lỗi nhầm từ chữ sang số.
        if char_list[i] == 'D': char_list[i] = '0'
        if char_list[i] == 'B': char_list[i] = '8'
        if char_list[i] == 'G': char_list[i] = '6'
        if char_list[i] == 'S': char_list[i] = '5'
        if char_list[i] == 'Z': char_list[i] = '2'
        if char_list[i] == 'V': char_list[i] = '7'  # <<< QUY TẮC MỚI

    return "".join(char_list)

def predict_char_hybrid(img):
    """
    Hybrid prediction: Luôn chạy cả 2 model và ưu tiên EasyOCR nếu đủ tự tin.
    """
    # Luôn chạy CNN trước
    cnn_char, cnn_conf = predict_char_cnn(img)

    # Nếu EasyOCR không có sẵn, trả về kết quả của CNN
    if not EASYOCR_AVAILABLE:
        return cnn_char, cnn_conf, "CNN"

    # Luôn chạy EasyOCR để có "ý kiến thứ hai"
    ocr_char, ocr_conf = predict_char_easyocr(img)

    # --- LOGIC QUYẾT ĐỊNH MỚI ---

    # Ưu tiên 1: Nếu EasyOCR rất tự tin (>= 0.7), hãy tin nó.
    # EasyOCR thường rất chính xác với các ký tự rõ ràng.
    if ocr_conf >= 0.7:
        return ocr_char, ocr_conf, "OCR (High Conf)"

    # Ưu tiên 2: Nếu cả hai đồng ý, kết hợp độ tin cậy.
    if cnn_char == ocr_char and cnn_conf > 0.5:
        return cnn_char, (cnn_conf + ocr_conf) / 2, "Both (Agreement)"

    # Ưu tiên 3: Nếu EasyOCR tự tin hơn CNN một cách đáng kể.
    if ocr_conf > cnn_conf * 1.2 and ocr_conf > 0.4:  # OCR hơn 20% và trên ngưỡng cơ bản
        return ocr_char, ocr_conf, "OCR (Stronger)"

    # Mặc định: Nếu không có trường hợp nào ở trên, quay về với CNN.
    return cnn_char, cnn_conf, "CNN (Default)"


# ===================== Plate detection =====================
def rotate_plate_auto(plate_img):
    if plate_img is None or plate_img.size == 0:
        return plate_img, 0

    h, w = plate_img.shape[:2]
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding for robust binarization
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)

    try:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return plate_img, 0

    # Filter out very small or very large contours that are unlikely to be the main plate
    min_contour_area = (h * w) * 0.1
    max_contour_area = (h * w) * 0.9

    filtered_contours = [cnt for cnt in contours if min_contour_area < cv2.contourArea(cnt) < max_contour_area]

    if not filtered_contours:
        return plate_img, 0

    largest_contour = max(filtered_contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    # Adjust angle to be in a more intuitive range (-45 to 45 degrees)
    if angle < -45:
        angle = 90 + angle

    # Only rotate if the angle is significant
    if abs(angle) < 1.0:  # Threshold for considering rotation
        print(f"  📐 Plate aligned (angle: {angle:.2f}°)")
        return plate_img, 0

    print(f"  🔄 Rotating plate: {angle:.2f}°")

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new dimensions to avoid cropping after rotation
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(plate_img, M, (new_w, new_h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated, angle


def detect_plate_advanced(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)  # Denoise while preserving edges
    edges = cv2.Canny(blur, 50, 200)  # Canny edge detection

    # Morphological operations to close gaps in edges and make contours more distinct
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(closed, kernel_dilate, iterations=1)

    try:
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        _, contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest first) to prioritize bigger regions
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]  # Consider top 50

    img_height, img_width = img.shape[:2]
    plate_candidates = []

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)  # Approximate polygon

        if len(approx) >= 4:  # Look for quadrilateral shapes
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            area = w * h

            # Filter based on common plate dimensions and size relative to image
            min_plate_area = (img_width * img_height) * 0.005  # Min 0.5% of image area
            max_plate_area = (img_width * img_height) * 0.98  # Max 80% of image area

            if (2.0 < aspect_ratio < 5.5 and area > min_plate_area and area < max_plate_area and w > 60 and h > 15):

                # === BƯỚC LỌC MÀU SẮC MỚI THÊM VÀO ===
                # Cắt vùng ảnh ứng viên ra để kiểm tra màu
                candidate_crop = img[y:y + h, x:x + w]

                # Chỉ thêm vào danh sách nếu nó vượt qua bộ lọc màu
                if is_plate_by_color(candidate_crop):
                    plate_candidates.append((x, y, w, h, area))


    if not plate_candidates:
        return None, None

    # Select the best candidate based on area (or a more complex scoring)
    x, y, w, h, _ = max(plate_candidates, key=lambda p: p[4])

    # Add a small padding to ensure the whole plate is captured
    pad = 5  # Pixels
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)

    plate_img = img[y:y + h, x:x + w]
    return plate_img, (x, y, w, h)


try:
    plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
    if plate_cascade.empty():
        raise IOError("Cannot load haarcascade")
    print("✅ Haar Cascade loaded")
except Exception as e:
    plate_cascade = None
    print(f"⚠️ Haar Cascade not found or loaded: {e}")


def detect_plate_haar(img):
    if plate_cascade is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast for Haar Cascade

    # Experiment with different scale factors to find plates of various sizes
    plates = plate_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,  # Default scaleFactor and minNeighbors
        minSize=(70, 20), maxSize=(800, 300)  # Refined min/max plate size
    )

    if len(plates) > 0:
        # Get the largest detected plate
        x, y, w, h = max(plates, key=lambda r: r[2] * r[3])

        # Add a small padding to ensure the whole plate is captured
        pad = 8
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)

        plate_img = img[y1:y2, x1:x2]
        return plate_img, (x1, y1, x2 - x1, y2 - y1)
    return None, None


# ===================== PERSPECTIVE CORRECTION =====================
def order_points(pts):
    # Khởi tạo một danh sách các tọa độ sẽ được sắp xếp
    # theo thứ tự: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # Tọa độ top-left sẽ có tổng (x+y) nhỏ nhất
    # Tọa độ bottom-right sẽ có tổng (x+y) lớn nhất
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Tọa độ top-right sẽ có hiệu (y-x) nhỏ nhất
    # Tọa độ bottom-left sẽ có hiệu (y-x) lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def correct_plate_perspective(plate_img):
    """
    Tìm 4 góc của biển số và biến đổi phối cảnh để làm phẳng nó.
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img  # Trả về ảnh gốc nếu không hợp lệ

    # Chuyển sang ảnh xám và làm mờ nhẹ để giảm nhiễu
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sử dụng ngưỡng thích ứng để làm nổi bật biển số
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    # Tìm các đường viền bên ngoài cùng
    try:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("  ⚠️ Perspective Correction: No contours found. Returning original plate.")
        return plate_img

    # Lọc và lấy đường viền lớn nhất
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    plate_contour = None

    # Tìm đường viền đầu tiên có 4 góc
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is None:
        print("  ⚠️ Perspective Correction: Could not find a 4-point contour. Returning original plate.")
        return plate_img  # Nếu không tìm thấy hình 4 cạnh, dùng ảnh gốc

    # Sắp xếp 4 điểm theo thứ tự chuẩn
    ordered_corners = order_points(plate_contour.reshape(4, 2))
    (tl, tr, br, bl) = ordered_corners

    # Tính chiều rộng mới của ảnh (khoảng cách euclid giữa 2 điểm top hoặc 2 điểm bottom)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Tính chiều cao mới của ảnh
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Tạo ma trận các điểm đến (ảnh phẳng)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Tính ma trận biến đổi phối cảnh và áp dụng nó
    M = cv2.getPerspectiveTransform(ordered_corners, dst)
    warped = cv2.warpPerspective(plate_img, M, (maxWidth, maxHeight))

    print("  ✨ Perspective correction applied successfully.")
    return warped


def is_plate_by_color(plate_img, threshold=0.3):
    """
    Kiểm tra xem một vùng ảnh có màu sắc giống biển số không (ít màu bão hòa).
    Biển số thường có màu trắng/đen/vàng, có độ bão hòa (saturation) thấp.
    Đèn hậu, logo màu... có độ bão hòa cao.
    """
    if plate_img is None or plate_img.size == 0:
        return False

    # Chuyển sang không gian màu HSV
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)

    # Lấy kênh Saturation (Độ bão hòa màu)
    # S channel range is [0, 255]
    s = hsv[:, :, 1]

    # Tính toán tỷ lệ các pixel có độ bão hòa cao
    # Coi các pixel có S > 80 là có màu sắc (không phải trắng/xám/đen)
    high_saturation_pixels = np.sum(s > 80)
    total_pixels = plate_img.shape[0] * plate_img.shape[1]

    saturation_ratio = high_saturation_pixels / total_pixels

    # Nếu tỷ lệ pixel có màu sắc quá cao (VD: > 30%), thì đây không phải biển số
    if saturation_ratio > threshold:
        # print(f"  - Rejected by color filter (Saturation: {saturation_ratio:.2f})")
        return False

    return True

def detect_and_extract_characters(plate_img):
    if plate_img is None or plate_img.size == 0:
        return [], None, None
    # Sử dụng hàm hiệu chỉnh phối cảnh mới thay vì chỉ xoay
    corrected_plate = correct_plate_perspective(plate_img)
    #rotated_plate, _ = rotate_plate_auto(plate_img)
    # Thay đổi kích thước biển số về một chiều cao cố định để ổn định việc lọc ký tự
    target_height = 100
    aspect_ratio = corrected_plate.shape[1] / corrected_plate.shape[0]
    target_width = int(target_height * aspect_ratio)
    corrected_plate = cv2.resize(corrected_plate, (target_width, target_height))

    # Sử dụng ảnh nhị phân đã được đảo ngược (ký tự trắng, nền đen) để tìm contour
    processed_plate_inv = preprocess_plate_binary(corrected_plate)

    if processed_plate_inv is None:
        return [], corrected_plate, None

    try:
        contours, _ = cv2.findContours(processed_plate_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        _, contours, _ = cv2.findContours(processed_plate_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_regions = []
    plate_h, plate_w = processed_plate_inv.shape

    # Các ngưỡng lọc ký tự (dựa trên tỷ lệ với chiều cao biển số)
    # Nới lỏng các điều kiện hơn
    min_char_h, max_char_h = 0.25 * plate_h, 0.9 * plate_h
    min_char_w, max_char_w = 0.03 * plate_w, 0.5 * plate_w

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # BỘ LỌC 1: Lọc theo kích thước cơ bản
        if not (min_char_h < h < max_char_h and min_char_w < w < max_char_w):
            continue

        aspect_ratio = w / float(h)

        # BỘ LỌC 2: Lọc dấu chấm và các nhiễu quá nhỏ/dẹt
        # Dấu chấm thường có tỷ lệ ~1 và diện tích nhỏ
        if cv2.contourArea(cnt) < (plate_h * 0.1) ** 2 and aspect_ratio < 1.3:
            continue

        # BỘ LỌC 3: Lọc theo tỷ lệ khung hình của ký tự - ĐÃ NỚI LỎNG
        if not (0.2 < aspect_ratio < 1.2):
            continue

        char_regions.append({'x': x, 'y': y, 'w': w, 'h': h, 'cy': y + h / 2})

    if not char_regions:
        return [], corrected_plate, None

    # LOGIC XỬ LÝ 2 DÒNG BẰNG K-MEANS CLUSTERING (thông minh và ổn định hơn)
    # Chỉ áp dụng khi có đủ ký tự để phân thành 2 dòng (ví dụ > 4)
    if len(char_regions) > 4:
        # Lấy tất cả các tọa độ y trung tâm
        y_centers = np.float32([char['cy'] for char in char_regions])

        # Áp dụng K-Means để gom thành 2 cụm (K=2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(y_centers, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Tách các ký tự thành 2 dòng dựa trên nhãn (label) từ K-Means
        line1 = [char for i, char in enumerate(char_regions) if labels[i] == 0]
        line2 = [char for i, char in enumerate(char_regions) if labels[i] == 1]

        # Sắp xếp 2 dòng theo thứ tự từ trên xuống dưới
        # Dòng nào có tọa độ y trung bình nhỏ hơn là dòng trên
        lines = sorted([line1, line2], key=lambda line: np.mean([char['cy'] for char in line]))
    else:
        # Nếu có quá ít ký tự, coi tất cả là 1 dòng
        lines = [char_regions]

    # Nhận diện ký tự theo từng dòng đã sắp xếp
    recognized_chars = []
    char_segmentation_img = corrected_plate.copy()

    for line in lines:
        # Sắp xếp các ký tự trong mỗi dòng theo thứ tự từ trái qua phải
        line_sorted = sorted(line, key=lambda c: c['x'])
        for char_info in line_sorted:
            x, y, w, h = char_info['x'], char_info['y'], char_info['w'], char_info['h']
            # Thêm một khoảng đệm nhỏ để mô hình CNN nhận diện tốt hơn
            pad = 2
            char_img = corrected_plate[max(0, y - pad):min(plate_h, y + h + pad),
                       max(0, x - pad):min(plate_w, x + w + pad)]

            char, conf, method = predict_char_hybrid(char_img)
            if char != '?':  # Chỉ thêm các ký tự nhận diện được
                recognized_chars.append((char, conf, method))

            # Vẽ hình chữ nhật và ký tự lên ảnh để debug
            cv2.rectangle(char_segmentation_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(char_segmentation_img, char, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    raw_chars = [char for char, _, _ in recognized_chars]
    raw_confidences = [conf for _, conf, _ in recognized_chars]

    return raw_chars, raw_confidences, char_segmentation_img
# ===================== Main Recognition Function =====================
def recognize_plate_from_image(image_path):
    print(f"\n--- Processing {os.path.basename(image_path)} ---")
    start_time = time.time()

    original_img = cv2.imread(image_path)
    if original_img is None:
        messagebox.showerror("Error", f"Could not load image: {image_path}")
        return None, None, None, None, None

    display_img = original_img.copy()

    plate_roi = None
    plate_bbox = None

    # Try Advanced detection first (usually better for complex backgrounds)
    plate_roi, plate_bbox = detect_plate_advanced(original_img)
    if plate_roi is not None:
        print("🔍 Plate detected using Advanced method.")
        cv2.rectangle(display_img, (plate_bbox[0], plate_bbox[1]),
                      (plate_bbox[0] + plate_bbox[2], plate_bbox[1] + plate_bbox[3]),
                      (255, 0, 0), 3)  # Blue bounding box for detected plate
    else:
        # Fallback to Haar Cascade if Advanced fails
        plate_roi, plate_bbox = detect_plate_haar(original_img)
        if plate_roi is not None:
            print("🔍 Plate detected using Haar Cascade method.")
            cv2.rectangle(display_img, (plate_bbox[0], plate_bbox[1]),
                          (plate_bbox[0] + plate_bbox[2], plate_bbox[1] + plate_bbox[3]),
                          (0, 0, 255), 3)  # Red bounding box for detected plate

    if plate_roi is None:
        messagebox.showwarning("Warning", "No license plate detected in the image.")
        return original_img, None, None, None, None  # Return original image and no results

    # If plate detected, proceed with character segmentation and recognition
    recognized_chars, confidences, char_segmentation_img = detect_and_extract_characters(plate_roi)

    plate_text = "".join(recognized_chars)
    plate_text = post_process_plate_text(plate_text)  # <<< Đảm bảo dòng này vẫn tồn tại
    print(f"  📝 Recognized characters after post-processing: {plate_text}")
    # Try to infer province code
    province = "N/A"
    if len(plate_text) >= 2 and plate_text[:2].isdigit():
        province_code = plate_text[:2]
        province = province_dict.get(province_code, "Unknown Province")
        print(f"  📍 Province code: {province_code} ({province})")
    elif len(plate_text) > 2 and plate_text[0].isdigit() and plate_text[
        1].isdigit():  # Handle cases like 51A-123.45 where 51 is the code
        province_code = plate_text[:2]
        province = province_dict.get(province_code, "Unknown Province")
        print(f"  📍 Province code (possible): {province_code} ({province})")

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"--- Processing complete in {processing_time:.2f} seconds ---")

    # Overlay recognized text and province on the original image for display
    if plate_bbox and recognized_chars:
        text_to_display = f"{plate_text} ({province})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        text_size = cv2.getTextSize(text_to_display, font, font_scale, font_thickness)[0]

        # Position text above the bounding box of the plate
        text_x = plate_bbox[0]
        text_y = plate_bbox[1] - 10 if plate_bbox[1] - 10 > text_size[1] else plate_bbox[1] + plate_bbox[3] + text_size[
            1] + 10

        # Ensure text is within image bounds
        text_y = max(20, min(text_y, original_img.shape[0] - 10))

        cv2.putText(display_img, text_to_display, (text_x, text_y),
                    font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

    # Return the processed image with bounding boxes and text, the raw plate image, the characters image, and the recognized text
    return display_img, plate_roi, char_segmentation_img, plate_text, province


# ===================== GUI Application =====================
class LicensePlateRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("License Plate Recognizer (Hybrid CNN + EasyOCR)")
        master.geometry("1200x800")  # Initial window size

        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.plate_roi_image = None
        self.char_seg_image = None
        self.recognized_text = ""
        self.province_info = ""

        # --- Frames ---
        self.input_frame = tk.Frame(master, bd=2, relief=tk.GROOVE)
        self.input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.display_frame = tk.Frame(master, bd=2, relief=tk.GROOVE)
        self.display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.output_frame = tk.Frame(master, bd=2, relief=tk.GROOVE)
        self.output_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # --- Input Frame Widgets ---
        self.load_button = tk.Button(self.input_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.path_label = tk.Label(self.input_frame, text="No image loaded", width=60, anchor="w")
        self.path_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.recognize_button = tk.Button(self.input_frame, text="Recognize Plate", command=self.recognize_plate_gui,
                                          state=tk.DISABLED)
        self.recognize_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # --- Display Frame Widgets ---
        # Using Canvas for image display allows for easy resizing and drawing
        self.canvas_width = 800
        self.canvas_height = 500
        self.image_canvas = tk.Canvas(self.display_frame, bg="lightgray", width=self.canvas_width,
                                      height=self.canvas_height)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Right Panel for Plate ROI and Characters ---
        self.right_panel = tk.Frame(self.display_frame, width=250)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        tk.Label(self.right_panel, text="Detected Plate").pack(pady=2)
        self.plate_roi_canvas = tk.Canvas(self.right_panel, bg="white", width=240, height=100, bd=1, relief=tk.SUNKEN)
        self.plate_roi_canvas.pack(pady=2)

        tk.Label(self.right_panel, text="Segmented Characters").pack(pady=2)
        self.char_seg_canvas = tk.Canvas(self.right_panel, bg="white", width=240, height=200, bd=1, relief=tk.SUNKEN)
        self.char_seg_canvas.pack(pady=2)

        # --- Output Frame Widgets ---
        self.result_label = tk.Label(self.output_frame, text="Recognized Plate: ", font=("Helvetica", 16, "bold"),
                                     anchor="w")
        self.result_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.province_label = tk.Label(self.output_frame, text="Province: ", font=("Helvetica", 14), anchor="w")
        self.province_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # --- Bind canvas resize event ---
        self.image_canvas.bind("<Configure>", self.on_canvas_resize)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        if file_path:
            self.image_path = file_path
            self.path_label.config(text=f"Loaded: {os.path.basename(self.image_path)}")
            self.recognize_button.config(state=tk.NORMAL)

            self.original_image = cv2.imread(self.image_path)
            self.display_image_on_canvas(self.original_image, self.image_canvas)

            # Clear previous results
            self.clear_results()

    def clear_results(self):
        self.result_label.config(text="Recognized Plate: ")
        self.province_label.config(text="Province: ")
        self.plate_roi_canvas.delete("all")
        self.char_seg_canvas.delete("all")
        self.plate_roi_image = None
        self.char_seg_image = None
        self.recognized_text = ""
        self.province_info = ""

    def recognize_plate_gui(self):
        if self.image_path:
            self.clear_results()
            self.recognize_button.config(state=tk.DISABLED, text="Processing...")
            self.master.update_idletasks()  # Update GUI to show button state change

            # Run recognition in a separate thread if it's too long
            # For simplicity, running in main thread for now.
            processed_img, plate_roi_img, char_seg_img, plate_text, province = recognize_plate_from_image(
                self.image_path)

            self.processed_image = processed_img
            self.plate_roi_image = plate_roi_img
            self.char_seg_image = char_seg_img
            self.recognized_text = plate_text
            self.province_info = province

            if self.processed_image is not None:
                self.display_image_on_canvas(self.processed_image, self.image_canvas)

            if self.plate_roi_image is not None:
                self.display_image_on_canvas(self.plate_roi_image, self.plate_roi_canvas, max_dim=240)

            if self.char_seg_image is not None:
                self.display_image_on_canvas(self.char_seg_image, self.char_seg_canvas, max_dim=240)

            self.result_label.config(text=f"Recognized Plate: {self.recognized_text}")
            self.province_label.config(text=f"Province: {self.province_info}")

            self.recognize_button.config(state=tk.NORMAL, text="Recognize Plate")
        else:
            messagebox.showwarning("Warning", "Please load an image first.")

    def display_image_on_canvas(self, cv_image, canvas_widget, max_dim=None):
        if cv_image is None:
            canvas_widget.delete("all")
            return

        # Convert OpenCV image to PIL Image
        # OpenCV reads as BGR, PIL expects RGB
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Resize image to fit canvas while maintaining aspect ratio
        canvas_width = canvas_widget.winfo_width()
        canvas_height = canvas_widget.winfo_height()

        if max_dim:  # For smaller canvases like plate_roi, char_seg
            target_width = max_dim
            target_height = max_dim
        else:  # For main image canvas
            target_width = canvas_width
            target_height = canvas_height

        if target_width <= 0 or target_height <= 0:  # Canvas might not be fully configured yet
            target_width = self.canvas_width  # Fallback to initial size
            target_height = self.canvas_height

        img_width, img_height = pil_img.size

        ratio_w = target_width / img_width
        ratio_h = target_height / img_height

        scale_factor = min(ratio_w, ratio_h)

        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)

        # Ensure minimum size to prevent issues if canvas is very small
        if new_width == 0 or new_height == 0:
            new_width = 1
            new_height = 1

        pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(pil_img)  # Keep a reference to prevent garbage collection

        canvas_widget.delete("all")
        # Center the image on the canvas
        x_center = (target_width - new_width) // 2
        y_center = (target_height - new_height) // 2
        canvas_widget.create_image(x_center, y_center, anchor=tk.NW, image=self.tk_image)
        canvas_widget.image = self.tk_image  # Store reference on the canvas itself

    def on_canvas_resize(self, event):
        # Redraw the main image when the canvas is resized
        if self.processed_image is not None:
            self.display_image_on_canvas(self.processed_image, self.image_canvas)
        elif self.original_image is not None:
            self.display_image_on_canvas(self.original_image, self.image_canvas)


# ===================== Run the App =====================
if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateRecognizerApp(root)
    root.mainloop()