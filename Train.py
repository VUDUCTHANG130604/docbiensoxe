import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import json
import os

# ================== CẤU HÌNH HUẤN LUYỆN ==================
data_dir = "Data"         # thư mục chứa các folder ký tự (0,1,...,A,B,...)
image_size = (28, 28)     # Kích thước ảnh đầu vào
batch_size = 32           # Kích thước batch
epochs = 50               # Số lượng epoch tối đa
validation_split = 0.2    # Tỷ lệ dữ liệu dành cho tập kiểm định

# Chọn loại mô hình (True cho CNN, False cho ANN)
use_cnn_model = True

# ================== KIỂM TRA THƯ MỤC DỮ LIỆU ==================
if not os.path.exists(data_dir):
    print(f"Lỗi: Thư mục dữ liệu '{data_dir}' không tồn tại. Vui lòng tạo và đặt dữ liệu vào đó.")
    exit()

# ================== TẠO DỮ LIỆU VỚI DATA AUGMENTATION ==================
print(f"\n⚙️ Cấu hình ImageDataGenerator...")
datagen = ImageDataGenerator(
    rescale=1./255,             # Chuẩn hóa giá trị pixel về [0, 1]
    validation_split=validation_split,
    rotation_range=5,           # Xoay ảnh ngẫu nhiên ±5 độ
    width_shift_range=0.1,      # Dịch chuyển ngang ảnh 10% chiều rộng
    height_shift_range=0.1,     # Dịch chuyển dọc ảnh 10% chiều cao
    zoom_range=0.1,             # Phóng/thu ảnh 10%
    shear_range=0.1,            # Biến dạng cắt 10% (có thể thêm)
    fill_mode="nearest"         # Cách điền các pixel mới sau biến đổi
)

# Tạo generator cho tập huấn luyện
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    color_mode="grayscale",     # ANN và CNN cho ảnh xám đều nhận 1 kênh
    batch_size=batch_size,
    class_mode="categorical",   # Phù hợp cho phân loại đa lớp
    subset="training",
    shuffle=True
)

# Tạo generator cho tập kiểm định
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False               # Không cần xáo trộn tập kiểm định
)

# Lấy số lượng lớp thực tế từ generator
num_classes = train_generator.num_classes
class_names = list(train_generator.class_indices.keys())
print(f"Đã phát hiện {num_classes} lớp: {class_names}")

# Xác định input_shape cho mô hình (kích thước ảnh, số kênh)
input_shape = (image_size[0], image_size[1], 1) # 1 kênh cho ảnh xám

# ================== XÂY DỰNG MÔ HÌNH ==================

def build_ann_model(input_shape, num_classes):
    """Xây dựng mô hình Artificial Neural Network (ANN)"""
    print("\n🏗️ Xây dựng mô hình ANN...")
    model = Sequential([
        # Flatten ảnh 28x28x1 thành vector 784 phần tử
        Flatten(input_shape=(input_shape[0], input_shape[1])),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax") # Output layer với softmax cho phân loại đa lớp
    ])
    return model

def build_cnn_model(input_shape, num_classes):
    """Xây dựng mô hình Convolutional Neural Network (CNN)"""
    print("\n🏗️ Xây dựng mô hình CNN...")
    model = Sequential([
        # Lớp Convolution đầu tiên
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25), # Thêm Dropout sau MaxPooling

        # Lớp Convolution thứ hai
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25), # Thêm Dropout sau MaxPooling

        # Flatten output của lớp Conv để đưa vào lớp Dense
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Dropout mạnh hơn cho lớp Dense cuối cùng
        Dense(num_classes, activation='softmax')
    ])
    return model

# Chọn mô hình dựa trên cấu hình
if use_cnn_model:
    model = build_cnn_model(input_shape, num_classes)
    model_type_name = "CNN"
else:
    model = build_ann_model(input_shape, num_classes)
    model_type_name = "ANN"

# Biên dịch mô hình
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# ================== CALLBACKS ==================
# Dừng sớm nếu val_loss không cải thiện sau 5 epoch và khôi phục trọng số tốt nhất
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Lưu mô hình tốt nhất dựa trên val_loss
model_checkpoint = ModelCheckpoint(
    f"best_{model_type_name}_model1.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint]

# ================== HUẤN LUYỆN MÔ HÌNH ==================
print(f"\n🚀 Bắt đầu huấn luyện mô hình {model_type_name}...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks
)
print(f"✅ Kết thúc huấn luyện mô hình {model_type_name}.")

# ================== LƯU MÔ HÌNH & NHÃN ==================
# Lưu mô hình cuối cùng (có thể không phải tốt nhất nếu EarlyStopping kích hoạt)
# Thường bạn sẽ dùng 'best_CNN_model.h5' hoặc 'best_ANN_model.h5' từ ModelCheckpoint
final_model_path = f"{model_type_name}_final_trained.h5"
model.save(final_model_path)
print(f"💾 Mô hình huấn luyện cuối cùng đã được lưu tại: {final_model_path}")

# Lưu mapping class → index để dễ dự đoán sau này
class_indices_path = "class_indices.json"
with open(class_indices_path, "w") as f:
    # Đảo ngược dictionary để có mapping từ index -> class name nếu cần
    # Hoặc lưu trực tiếp như bạn đã làm: class name -> index
    json.dump(train_generator.class_indices, f)
print(f"💾 class_indices.json đã được lưu.")

print(f"\n✨ Quá trình huấn luyện mô hình {model_type_name} hoàn tất.")

# ================== VẼ BIỂU ĐỒ TRAINING ==================
plt.figure(figsize=(12,5))

# Biểu đồ Accuracy
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title(f"{model_type_name} Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Biểu đồ Loss
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title(f"{model_type_name} Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()