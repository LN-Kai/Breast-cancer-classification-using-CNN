import os
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class BreastCancerClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phân loại ảnh ung thư vú")
        self.root.geometry("900x600")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")
        
        self.model = None
        self.class_names = ["benign", "malignant", "normal"]
        self.image_path = None
        self.img_size = (224, 224)
        
        self.setup_ui()
        self.load_model_from_file()
    
    def setup_ui(self):
        # Tạo frame chính
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Tiêu đề
        title_label = tk.Label(
            main_frame, 
            text="Ứng dụng phân loại ảnh ung thư vú", 
            font=("Arial", 18, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50"
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Frame bên trái: Hiển thị ảnh
        left_frame = tk.Frame(main_frame, bg="#f0f0f0")
        left_frame.grid(row=1, column=0, sticky="nsew", padx=10)
        
        # Frame hiển thị ảnh
        self.image_frame = tk.Frame(left_frame, bg="white", width=400, height=400)
        self.image_frame.pack(pady=20)
        
        # Label hiển thị ảnh
        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Frame bên phải: Kết quả và nút
        right_frame = tk.Frame(main_frame, bg="#f0f0f0")
        right_frame.grid(row=1, column=1, sticky="nsew", padx=10)
        
        # Frame nút
        button_frame = tk.Frame(right_frame, bg="#f0f0f0")
        button_frame.pack(pady=10)
        
        # Nút chọn ảnh
        self.select_button = tk.Button(
            button_frame,
            text="Chọn ảnh",
            command=self.select_image,
            bg="#3498db",
            fg="white",
            font=("Arial", 12),
            width=15,
            relief=tk.FLAT
        )
        self.select_button.grid(row=0, column=0, padx=10)
        
        # Nút phân loại
        self.classify_button = tk.Button(
            button_frame,
            text="Phân loại",
            command=self.classify_image,
            bg="#2ecc71",
            fg="white",
            font=("Arial", 12),
            width=15,
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.classify_button.grid(row=0, column=1, padx=10)
        
        # Nút thoát
        self.exit_button = tk.Button(
            button_frame,
            text="Thoát",
            command=self.root.quit,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 12),
            width=15,
            relief=tk.FLAT
        )
        self.exit_button.grid(row=0, column=2, padx=10)
        
        # Frame kết quả
        result_frame = tk.Frame(right_frame, bg="#f0f0f0")
        result_frame.pack(pady=20, fill=tk.X)
        
        # Label cho kết quả
        self.result_label = tk.Label(
            result_frame,
            text="Kết quả phân loại sẽ hiển thị ở đây",
            font=("Arial", 14),
            bg="#f0f0f0",
            fg="#2c3e50"
        )
        self.result_label.pack()
        
        # Label chi tiết kết quả
        self.detail_label = tk.Label(
            result_frame,
            text="",
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="#7f8c8d",
            justify=tk.LEFT
        )
        self.detail_label.pack(pady=10)
        
        # Cấu hình grid cho main_frame
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        
        # Thanh trạng thái
        self.status_bar = tk.Label(
            self.root, 
            text="Sẵn sàng", 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Các phương thức giữ nguyên
    def load_model_from_file(self):
        try:
            self.status_bar.config(text="Đang tải mô hình...")
            model_path = 'final_model.keras'
            if not os.path.exists(model_path):
                model_path = filedialog.askopenfilename(
                    title="Chọn file mô hình",
                    filetypes=[("Keras Model", "*.keras"), ("HDF5 Model", "*.h5"), ("All Files", "*.*")]
                )
                if not model_path:
                    messagebox.showerror("Lỗi", "Không tìm thấy mô hình!")
                    self.root.quit()
                    return
            self.model = load_model(model_path)
            self.status_bar.config(text=f"Đã tải mô hình từ {model_path}")
            messagebox.showinfo("Thông báo", "Đã tải mô hình thành công!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải mô hình: {str(e)}")
            self.status_bar.config(text="Lỗi khi tải mô hình")
            
    def select_image(self):
        image_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")]
        )
        if image_path:
            self.image_path = image_path
            self.display_image(image_path)
            self.classify_button.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Đã chọn ảnh: {os.path.basename(image_path)}")
            self.result_label.config(text="Kết quả phân loại sẽ hiển thị ở đây")
            self.detail_label.config(text="")
    
    def display_image(self, image_path):
        image = Image.open(image_path)
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo
    
    def preprocess_image(self, image_path):
        try:
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            return img_array
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xử lý ảnh: {str(e)}")
            return None
    
    def classify_image(self):
        if self.image_path and self.model:
            try:
                self.status_bar.config(text="Đang phân loại ảnh...")
                processed_image = self.preprocess_image(self.image_path)
                if processed_image is None:
                    return
                predictions = self.model.predict(processed_image)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = self.class_names[predicted_class_index]
                confidence = predictions[0][predicted_class_index] * 100
                result_text = f"Kết quả: {predicted_class}"
                if predicted_class == "Normal":
                    result_color = "#2ecc71"
                    result_desc = "Không phát hiện dấu hiệu bất thường"
                elif predicted_class == "Benign":
                    result_color = "#f39c12"
                    result_desc = "Phát hiện khối u lành tính"
                else:
                    result_color = "#e74c3c"
                    result_desc = "Phát hiện khối u ác tính - cần kiểm tra ngay"
                self.result_label.config(text=result_text, fg=result_color, font=("Arial", 16, "bold"))
                detail_text = f"{result_desc}\n\nĐộ tin cậy: {confidence:.2f}%\n\n"
                detail_text += "Chi tiết dự đoán:\n"
                for i, class_name in enumerate(self.class_names):
                    detail_text += f"- {class_name}: {predictions[0][i]*100:.2f}%\n"
                self.detail_label.config(text=detail_text)
                self.status_bar.config(text=f"Đã phân loại ảnh: {predicted_class}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi phân loại ảnh: {str(e)}")
                self.status_bar.config(text="Lỗi khi phân loại ảnh")
        else:
            if not self.image_path:
                messagebox.showinfo("Thông báo", "Vui lòng chọn ảnh trước!")
            elif not self.model:
                messagebox.showinfo("Thông báo", "Không thể tải mô hình!")

def main():
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"Đã tìm thấy GPU: {physical_devices[0]}")
    except:
        print("Không tìm thấy GPU hoặc lỗi khi thiết lập GPU")
    root = tk.Tk()
    app = BreastCancerClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()