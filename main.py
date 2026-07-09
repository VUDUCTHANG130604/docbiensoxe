import customtkinter as ctk
from PIL import Image
from tkinter import filedialog, messagebox
import cv2
import os
import threading

# Import các logic
from db import db_helper
from processing import detect  # Import file detect.py vừa tạo

# Khởi tạo database khi ứng dụng chạy
db_helper.create_table()


class PlateRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- CONFIG ---
        self.title("NHẬN DẠNG BIỂN SỐ XE")
        self.geometry("1200x750")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Kích thước cố định cho ảnh hiển thị
        self.FIXED_W = 800
        self.FIXED_H = 500

        # Load model ngầm
        self.after(100, lambda: threading.Thread(target=detect.load_models, daemon=True).start())

        # State
        self.current_image_path = None
        self.recognized_plate = ""
        self.plate_img_data = None

        # === Layout ===
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)

        # === LEFT FRAME: ORIGINAL IMAGE ===
        self.left_frame = ctk.CTkFrame(self, fg_color="#2A2D2E", corner_radius=10)
        self.left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.left_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self.left_frame, text="📸 ẢNH HIỂN THỊ",
            font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, pady=(15, 5))

        self.image_canvas = ctk.CTkLabel(
            self.left_frame,
            text="Vui lòng tải ảnh...",
            fg_color="#1F1F1F",
            text_color="#AAAAAA",
            corner_radius=8,
            width=self.FIXED_W,
            height=self.FIXED_H
        )
        self.image_canvas.grid(row=1, column=0, padx=15, pady=15)

        # === RIGHT FRAME: CONTROL ===
        self.right_frame = ctk.CTkFrame(self, fg_color="#3B3E3F", corner_radius=10)
        self.right_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        ctk.CTkLabel(
            self.right_frame,
            text="⚙️ BẢNG ĐIỀU KHIỂN",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(pady=(20, 10))

        # Upload button
        self.upload_button = ctk.CTkButton(
            self.right_frame, text="📂 TẢI ẢNH LÊN",
            command=self.load_image, fg_color="#2980B9"
        )
        self.upload_button.pack(padx=30, pady=10, fill="x")

        # Detect button
        self.detect_button = ctk.CTkButton(
            self.right_frame, text="🚀 NHẬN DIỆN",
            command=self.run_recognition,
            state="disabled",
            fg_color="#27AE60"
        )
        self.detect_button.pack(padx=30, pady=10, fill="x")

        # Progress
        self.progress_bar = ctk.CTkProgressBar(self.right_frame, mode="indeterminate", width=250)
        self.progress_bar.pack_forget()

        # Result text
        self.result_frame = ctk.CTkFrame(self.right_frame, fg_color="#2B2B2B")
        self.result_frame.pack(padx=30, pady=20, fill="x")

        ctk.CTkLabel(self.result_frame, text="KẾT QUẢ:").pack(pady=(10, 0))
        self.result_label = ctk.CTkLabel(
            self.result_frame, text="---",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color="#E74C3C"
        )
        self.result_label.pack(pady=(0, 10))

        # Plate crop image
        ctk.CTkLabel(self.right_frame, text="🔍 Ảnh biển số:").pack(pady=(10, 0))
        self.plate_canvas = ctk.CTkLabel(
            self.right_frame, fg_color="#2B2B2B",
            height=100, corner_radius=8
        )
        self.plate_canvas.pack(padx=30, pady=5, fill="x")

        # Save button
        self.save_button = ctk.CTkButton(
            self.right_frame, text="💾 LƯU KẾT QUẢ",
            command=self.save_plate,
            state="disabled",
            fg_color="#D35400"
        )
        self.save_button.pack(padx=30, pady=20, fill="x")

        # Reset button
        ctk.CTkButton(
            self.right_frame, text="🔄 LÀM MỚI",
            command=self.reset_app,
            fg_color="#7F8C8D"
        ).pack(padx=30, pady=5, fill="x")

    # =========================
    #     IMAGE DISPLAY
    # =========================

    def display_image(self, ctk_label, img_data, is_plate=False):
        """Resize ảnh về 1 kích thước cố định"""
        if img_data is None:
            ctk_label.configure(image=None, text="Không có ảnh")
            return

        rgb_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        if is_plate:
            target_h = 100
            ratio = target_h / pil_img.height
            target_w = int(pil_img.width * ratio)
            if target_w > 300:
                target_w = 300
        else:
            # === SIZE CỐ ĐỊNH CHO MỌI ẢNH ===
            target_w = self.FIXED_W
            target_h = self.FIXED_H

        ctk_img = ctk.CTkImage(
            light_image=pil_img,
            dark_image=pil_img,
            size=(target_w, target_h)
        )
        ctk_label.configure(image=ctk_img, text="")
        ctk_label.image = ctk_img

    # =========================

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh xe",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        self.reset_app(full_reset=False)
        self.current_image_path = file_path

        img = cv2.imread(file_path)
        if img is not None:
            self.display_image(self.image_canvas, img)
            self.detect_button.configure(state="normal")
        else:
            messagebox.showerror("Lỗi", "Không đọc được ảnh!")

    # =========================

    def run_recognition(self):
        if not self.current_image_path:
            return

        self.detect_button.configure(state="disabled", text="⏳ ĐANG XỬ LÝ...")
        self.progress_bar.pack(pady=5)
        self.progress_bar.start()

        threading.Thread(target=self._process_thread).start()

    def _process_thread(self):
        try:
            text, plate_crop, img_with_box = detect.process_image_and_recognize(self.current_image_path)
            self.after(0, lambda: self._update_ui_result(text, plate_crop, img_with_box))
        except Exception as e:
            print(e)
            self.after(0, lambda: messagebox.showerror("Lỗi", str(e)))

    # =========================

    def _update_ui_result(self, text, plate_crop, img_with_box):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.detect_button.configure(state="normal", text="🚀 NHẬN DIỆN")

        self.recognized_plate = text

        # Ảnh có bounding box → resize theo FIXED_W, FIXED_H
        if img_with_box is not None:
            self.display_image(self.image_canvas, img_with_box)

        # Ảnh biển số
        if plate_crop is not None:
            self.display_image(self.plate_canvas, plate_crop, is_plate=True)

        # Hiển thị kết quả
        self.result_label.configure(text=text)

        if "LỖI" in text:
            self.result_label.configure(text_color="#E74C3C")
            self.save_button.configure(state="disabled")
        else:
            self.result_label.configure(text_color="#2ECC71")
            self.save_button.configure(state="normal")

    # =========================

    def save_plate(self):
        if self.recognized_plate and "LỖI" not in self.recognized_plate:
            if db_helper.save_plate_number(self.recognized_plate):
                messagebox.showinfo("OK", "Đã lưu!")
                self.save_button.configure(state="disabled", text="✅ ĐÃ LƯU")

    # =========================

    def reset_app(self, full_reset=True):
        self.recognized_plate = ""
        self.result_label.configure(text="---", text_color="#E74C3C")
        self.plate_canvas.configure(image=None, text="")
        self.save_button.configure(state="disabled", text="💾 LƯU KẾT QUẢ")

        if full_reset:
            self.current_image_path = None
            self.image_canvas.configure(image=None, text="Vui lòng tải ảnh...")
            self.detect_button.configure(state="disabled")


# --- RUN ---
if __name__ == "__main__":
    app = PlateRecognitionApp()
    app.mainloop()
