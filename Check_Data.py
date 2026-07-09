import matplotlib.pyplot as plt
import os
import random
import cv2
import numpy as np

# Đường dẫn thư mục dữ liệu
data_dir = "Data"

if not os.path.exists(data_dir):
    print(f"❌ Không tìm thấy thư mục '{data_dir}'")
    exit()

# Lấy danh sách các lớp (tên thư mục con)
classes = sorted(os.listdir(data_dir))
print(f"📂 Tìm thấy {len(classes)} lớp ký tự: {classes}")

# Tạo lưới hiển thị 5x5
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle(f"Kiểm tra ngẫu nhiên dữ liệu trong '{data_dir}'", fontsize=16)

# Duyệt qua các ô để vẽ ảnh
for i, ax in enumerate(axes.flat):
    # Chọn ngẫu nhiên 1 lớp
    random_class = random.choice(classes)
    class_path = os.path.join(data_dir, random_class)

    # Lấy danh sách ảnh trong lớp đó
    images = os.listdir(class_path)
    if not images:
        continue

    # Chọn ngẫu nhiên 1 ảnh
    random_image = random.choice(images)
    img_path = os.path.join(class_path, random_image)

    # Đọc ảnh
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        # Resize về 28x28 để xem giống như model nhìn thấy
        img = cv2.resize(img, (28, 28))

        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {random_class}", color="green", fontsize=10)
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, "Error", ha='center')

plt.tight_layout()
plt.show()