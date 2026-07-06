import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Reshape
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ ĐẦU VÀO
# ==========================================
DATASET_PATH = "Dataset"
SR = 16000
DURATION = 3
N_MELS = 128
MAX_TIME_STEPS = 94 

# Biến cờ để chỉ lưu 1 ảnh mẫu duy nhất cho mỗi lệnh (tránh tạo hàng trăm ảnh rác)
saved_sample_images = []

def extract_mel_spectrogram(file_path, label):
    """
    Chuyển file âm thanh thành ma trận Mel-Spectrogram.
    Đồng thời xuất ra 1 ảnh mẫu .png cho mỗi nhãn để làm báo cáo.
    """
    global saved_sample_images
    y, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    
    if len(y) < SR * DURATION:
        y = np.pad(y, (0, SR * DURATION - len(y)))
        
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # BỔ SUNG: VẼ VÀ LƯU ẢNH MẪU ĐỂ LÀM BÁO CÁO KHOA HỌC
    if label not in saved_sample_images:
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(mel_db, sr=SR, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-Spectrogram - Lệnh: {label}')
        plt.tight_layout()
        
        # Lưu ảnh vào thư mục hiện tại
        img_name = f"sample_spectrogram_{label}.png"
        plt.savefig(img_name)
        plt.close()
        saved_sample_images.append(label)
        print(f"📸 Đã lưu ảnh quang phổ mẫu: {img_name}")
    
    return mel_db.T 

# ==========================================
# 2. LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
print("🔄 Đang xử lý âm thanh thành ảnh quang phổ (Mel-Spectrogram)...")
X_data = []
y_label = []
labels = os.listdir(DATASET_PATH)
label_to_index = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder): continue
        
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            file_path = os.path.join(folder, file)
            # Trích xuất đặc trưng và vẽ ảnh mẫu
            spectrogram = extract_mel_spectrogram(file_path, label)
            X_data.append(spectrogram)
            y_label.append(label_to_index[label])

X_data = np.array(X_data)[..., np.newaxis] 
y_label = to_categorical(np.array(y_label), num_classes=len(labels))

X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.2, random_state=42)
print(f"✅ Dữ liệu sẵn sàng! Tổng số mẫu: {len(X_data)}")

# ==========================================
# 3. XÂY DỰNG KIẾN TRÚC CNN-LSTM
# ==========================================
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(MAX_TIME_STEPS, N_MELS, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Reshape((-1, 30 * 64))) 
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(len(labels), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 4. TIẾN HÀNH HUẤN LUYỆN
# ==========================================
print("\n🚀 BẮT ĐẦU HUẤN LUYỆN MẠNG NƠ-RON...")
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Lưu mô hình
model.save("voice_controller_cnn_lstm.h5")
print("\n🎉 ĐÃ LƯU MÔ HÌNH THÀNH CÔNG TẠI: voice_controller_cnn_lstm.h5")

with open("labels.txt", "w") as f:
    for label in labels:
        f.write(f"{label}\n")

# ==========================================
# 5. BỔ SUNG: VẼ BIỂU ĐỒ ĐÁNH GIÁ (TRAINING METRICS)
# ==========================================
print("📊 Đang xuất biểu đồ kết quả huấn luyện...")
plt.figure(figsize=(12, 5))

# Biểu đồ Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Tập Huấn luyện (Train)')
plt.plot(history.history['val_accuracy'], label='Tập Kiểm thử (Validation)')
plt.title('Độ chính xác của Mô hình (Model Accuracy)')
plt.xlabel('Vòng lặp (Epoch)')
plt.ylabel('Độ chính xác')
plt.legend()

# Biểu đồ Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Tập Huấn luyện (Train)')
plt.plot(history.history['val_loss'], label='Tập Kiểm thử (Validation)')
plt.title('Hàm Mất mát (Model Loss)')
plt.xlabel('Vòng lặp (Epoch)')
plt.ylabel('Giá trị Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
print("📸 Đã lưu biểu đồ huấn luyện: training_metrics.png")
plt.show() # Tự động mở cửa sổ bật lên để bạn xem ngay biểu đồ