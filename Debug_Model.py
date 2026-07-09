import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
import urllib.request
import tkinter as tk
from tkinter import filedialog

# =============================================================================
# CẤU HÌNH & CHUẨN BỊ
# =============================================================================

# Đổi tên file này nếu model của bạn tên khác
model_path = "CNN_final_trained.h5"
class_indices_path = "class_indices.json"
cascade_filename = "haarcascade_russian_plate_number.xml"


# 1. Tải Haar Cascade an toàn
def download_cascade():
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml"
    print(f"⬇️ Đang tải {cascade_filename} từ GitHub...")
    try:
        urllib.request.urlretrieve(url, cascade_filename)
        # Kiểm tra dung lượng file sau khi tải
        if os.path.getsize(cascade_filename) < 1000:  # File XML chuẩn phải > 1KB
            print("❌ Tải lỗi: File quá nhỏ hoặc rỗng.")
            os.remove(cascade_filename)
            return False
        print("✅ Đã tải xong file Haar Cascade!")
        return True
    except Exception as e:
        print(f"❌ Lỗi tải file: {e}")
        return False


# Kiểm tra nếu file chưa tồn tại hoặc bị lỗi (0 byte)
if not os.path.exists(cascade_filename) or os.path.getsize(cascade_filename) == 0:
    download_cascade()

# Khởi tạo Haar Cascade với Try-Catch để tránh crash
plate_cascade = None
try:
    if os.path.exists(cascade_filename):
        plate_cascade = cv2.CascadeClassifier(cascade_filename)
        if plate_cascade.empty():
            print("⚠️ File xml bị lỗi định dạng. Đang thử tải lại...")
            if download_cascade():
                plate_cascade = cv2.CascadeClassifier(cascade_filename)
            else:
                print("⚠️ Không thể tải file xml. Hãy tải thủ công!")
                plate_cascade = None
    else:
        print("⚠️ Không tìm thấy file haarcascade. Hãy tải thủ công!")
except Exception as e:
    print(f"❌ Lỗi khởi tạo Haar Cascade: {e}")
    plate_cascade = None

if plate_cascade is None:
    print("\n⚠️ QUAN TRỌNG: Bạn cần file 'haarcascade_russian_plate_number.xml' để chạy.")
    print("👉 Hãy tải thủ công tại: https://github.com/opencv/opencv/tree/master/data/haarcascades")
    print("   Sau đó copy vào thư mục dự án.\n")

# 2. Load Model
if not os.path.exists(model_path):
    # Thử tìm các tên khác
    alternatives = ["best_cnn_model.h5", "ANN_final_trained.h5", "best_ann_model.h5"]
    found = False
    for alt in alternatives:
        if os.path.exists(alt):
            model_path = alt
            found = True
            break
    if not found:
        print(f"❌ Không tìm thấy model nào! Hãy đảm bảo file .h5 nằm cùng thư mục.")
        exit()

print(f"🔄 Đang tải model {model_path}...")
model = tf.keras.models.load_model(model_path)

# 3. Load Labels
characters = list("0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ")
if os.path.exists(class_indices_path):
    with open(class_indices_path, "r") as f:
        indices = json.load(f)
        idx_map = {v: k for k, v in indices.items()}
        characters = [idx_map[i] for i in sorted(idx_map.keys())]


# =============================================================================
# CÁC HÀM XỬ LÝ (MÔ PHỎNG TEST.PY)
# =============================================================================

def detect_plate_debug(img):
    """Tìm và cắt biển số để debug chính xác hơn"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Thử Haar Cascade
    if plate_cascade is not None and not plate_cascade.empty():
        plates = plate_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
        if len(plates) > 0:
            x, y, w, h = max(plates, key=lambda p: p[2] * p[3])
            pad = 5
            x, y = max(0, x - pad), max(0, y - pad)
            w, h = min(img.shape[1] - x, w + 2 * pad), min(img.shape[0] - y, h + 2 * pad)
            return img[y:y + h, x:x + w]

    # 2. Thử Contour đơn giản (nếu Haar trượt hoặc chưa load được)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 30, 200)
    cnts, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 1000 and 1.5 < w / h < 5:  # Biển dài
                return img[y:y + h, x:x + w]
            if w * h > 1000 and 0.8 < w / h < 1.5:  # Biển vuông
                return img[y:y + h, x:x + w]

    return img  # Trả về ảnh gốc nếu không tìm thấy (để cắt đại)


def preprocess_debug(img):
    """Tạo ra 2 phiên bản: Chữ Trắng và Chữ Đen"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Kiểu 1: Chữ Trắng / Nền Đen (INV)
    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Kiểu 2: Chữ Đen / Nền Trắng (NORM)
    _, bin_norm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return gray, bin_inv, bin_norm


def prepare_for_model(binary_img, is_inv=True):
    """Đóng gói ảnh để đưa vào model"""
    h, w = binary_img.shape
    size = max(h, w) + 4

    # Chọn màu nền phù hợp với loại ảnh
    if is_inv:
        bg_color = 0  # Nền đen cho ảnh chữ trắng
    else:
        bg_color = 255  # Nền trắng cho ảnh chữ đen

    square = np.full((size, size), bg_color, np.uint8)

    x_pos = (size - w) // 2
    y_pos = (size - h) // 2
    square[y_pos:y_pos + h, x_pos:x_pos + w] = binary_img

    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    norm = resized.astype("float32") / 255.0
    return resized, norm.reshape(1, 28, 28, 1)


# =============================================================================
# CHẠY CHƯƠNG TRÌNH
# =============================================================================

root = tk.Tk()
root.withdraw()
print("👉 Hãy chọn file ảnh BIỂN SỐ XE...")
file_path = filedialog.askopenfilename()

if not file_path: exit()
img_org = cv2.imread(file_path)
if img_org is None:
    print("❌ Không đọc được ảnh.")
    exit()

# 1. Tìm biển số trước
print("🔍 Đang tìm biển số...")
plate_img = detect_plate_debug(img_org)

# 2. Cắt ký tự từ biển số
gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
thresh_plate = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
cnts, _ = cv2.findContours(thresh_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

char_crops = []
# Lọc contour để lấy ký tự
h_p, w_p = plate_img.shape[:2]
cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])  # Sort trái qua phải

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    # Logic lọc cơ bản
    if 0.2 < h / h_p < 0.95 and 100 < w * h < (w_p * h_p) / 5:
        char_crops.append(plate_img[y:y + h, x:x + w])

if not char_crops:
    print("⚠️ Không tìm thấy ký tự rõ ràng, sẽ cắt vùng giữa ảnh để test...")
    h, w = plate_img.shape[:2]
    char_crops.append(plate_img[h // 4:3 * h // 4, w // 4:3 * w // 4])

# Chỉ lấy tối đa 6 ký tự để hiển thị cho đẹp
char_crops = char_crops[:6]

# 3. Hiển thị bảng so sánh
plt.figure(figsize=(16, 9))
plt.suptitle(f"DEBUG MODEL (Ảnh: {os.path.basename(file_path)})", fontsize=16, color='blue', fontweight='bold')

rows = 3
cols = len(char_crops)

for i, char_img in enumerate(char_crops):
    gray, bin_inv, bin_norm = preprocess_debug(char_img)

    # Dự đoán Kiểu 1: Trắng/Đen
    vis_inv, inp_inv = prepare_for_model(bin_inv, is_inv=True)
    pred_inv = model.predict(inp_inv, verbose=0)
    res_inv = characters[np.argmax(pred_inv)]
    conf_inv = np.max(pred_inv) * 100

    # Dự đoán Kiểu 2: Đen/Trắng
    vis_norm, inp_norm = prepare_for_model(bin_norm, is_inv=False)
    pred_norm = model.predict(inp_norm, verbose=0)
    res_norm = characters[np.argmax(pred_norm)]
    conf_norm = np.max(pred_norm) * 100

    # DÒNG 1: ẢNH CẮT TỪ BIỂN SỐ
    ax1 = plt.subplot(rows, cols, i + 1)
    ax1.imshow(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB))
    if i == 0: ax1.set_ylabel("Ảnh gốc", fontsize=12, fontweight='bold')
    ax1.set_title(f"Char {i + 1}")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # DÒNG 2: MODEL NHÌN KIỂU TRẮNG/ĐEN
    ax2 = plt.subplot(rows, cols, i + 1 + cols)
    ax2.imshow(vis_inv, cmap='gray')
    if i == 0: ax2.set_ylabel("INPUT: TRẮNG/ĐEN", fontsize=12, fontweight='bold', color='purple')

    color = "green" if conf_inv > 80 else "red"
    ax2.set_xlabel(f"Đoán: {res_inv}\n{conf_inv:.1f}%", color=color, fontsize=12, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # DÒNG 3: MODEL NHÌN KIỂU ĐEN/TRẮNG
    ax3 = plt.subplot(rows, cols, i + 1 + cols * 2)
    ax3.imshow(vis_norm, cmap='gray')
    if i == 0: ax3.set_ylabel("INPUT: ĐEN/TRẮNG", fontsize=12, fontweight='bold', color='orange')

    color = "green" if conf_norm > 80 else "red"
    ax3.set_xlabel(f"Đoán: {res_norm}\n{conf_norm:.1f}%", color=color, fontsize=12, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])

plt.tight_layout()
plt.show()