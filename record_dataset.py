import sounddevice as sd
import soundfile as sf
import os
import time

# ==========================================
# CẤU HÌNH THÔNG SỐ AUDIO
# ==========================================
SR = 16000        # Tần số lấy mẫu 16kHz (chuẩn của giọng nói)
DURATION = 3.0    # 3 giây mỗi mẫu (khớp với độ trễ của IDE)
CHANNELS = 1      # Thu âm đơn sắc (Mono)

def record_samples(label_name, num_samples=15):
    """
    Hàm tự động tạo thư mục và thu âm liên tục nhiều mẫu cho một nhãn.
    """
    # Tạo thư mục cho nhãn nếu chưa có
    folder_path = os.path.join("Dataset", label_name)
    os.makedirs(folder_path, exist_ok=True)
    
    print(f"\n🚀 CHUẨN BỊ THU ÂM CHO LỆNH: '{label_name}'")
    print(f"📁 Dữ liệu sẽ lưu tại: {folder_path}")
    print("-" * 50)
    
    for i in range(num_samples):
        input(f"👉 Nhấn [ENTER] để bắt đầu thu mẫu {i+1}/{num_samples} (Nói sau khi thấy chữ ĐANG THU ÂM)...")
        
        print("🎙️ ĐANG THU ÂM (3 giây)...")
        # Gọi mic thu âm
        audio_data = sd.rec(int(DURATION * SR), samplerate=SR, channels=CHANNELS)
        sd.wait() # Chờ đủ 3 giây
        
        # Lưu file .wav
        file_name = f"sample_{i+1:03d}.wav"
        file_path = os.path.join(folder_path, file_name)
        sf.write(file_path, audio_data, SR)
        
        print(f"✅ Đã lưu thành công: {file_name}\n")

if __name__ == "__main__":
    # Danh sách các lệnh bạn muốn train (Bạn có thể thêm bớt tùy ý)
    commands = [
        "tao_file",
        "mo_terminal",
        "chay_code",
        "format_code",
        "che_do_toi"
    ]
    
    print("DANH SÁCH LỆNH CẦN THU ÂM:")
    for idx, cmd in enumerate(commands):
        print(f"{idx + 1}. {cmd}")
        
    choice = int(input("\nNhập số thứ tự lệnh bạn muốn thu âm bây giờ: ")) - 1
    selected_command = commands[choice]
    
    # Bạn nên thu khoảng 30-50 mẫu cho mỗi lệnh để mô hình không bị Underfitting
    record_samples(selected_command, num_samples=30)