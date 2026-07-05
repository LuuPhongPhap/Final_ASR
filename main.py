from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import whisper

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH PHOWHISPER
# ==========================================
print("🔥 Đang tải PhoWhisper bản SMALL trên CPU...")
model = whisper.load_model("small")  
print("✅ Tải mô hình hoàn tất!")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. XỬ LÝ LỆNH THỦ CÔNG (RULE-BASED NLP)
# ==========================================
def get_intent_from_text(text):
    """
    Hàm phân tích câu lệnh thủ công bằng if-else.
    Ổn định 100%, chạy offline, phản hồi tức thì.
    """
    text = text.lower()
    
    # 1. TẠO FILE
    if ("tạo" in text and "file" in text) or "thêm file" in text or "new file" in text or "hậu phái bớ" in text or "tao phai" in text:
        return "workbench.action.files.newUntitledFile"
        
    # 2. LƯU FILE
    elif ("lưu" in text and "file" in text) or "lưu lại" in text or "save" in text:
        return "workbench.action.files.save"
        
    # 3. ĐÓNG TAB/FILE
    elif ("đóng" in text and ("tab" in text or "file" in text)) or "tắt" in text:
        return "workbench.action.closeActiveEditor"
        
    # 4. CHẠY CODE
    elif "chạy code" in text or "run code" in text or "răn cốt" in text or "chạy cốt" in text or "răn" in text:
        return "code-runner.run" 
        
    # 5. MỞ TERMINAL
    elif "mở terminal" in text or "bật terminal" in text or "tơ mi nồ" in text or "tôi mi nồ" in text or "cho mi nồ" in text or "omega" in text or "tơ me lo" in text:
        return "workbench.action.terminal.toggleTerminal"
        
    # 6. XÓA TERMINAL
    elif "xóa terminal" in text or "clear terminal" in text or "cờ lia" in text or "xóa tơ mi nồ" in text:
        return "workbench.action.terminal.clear"
        
    # 7. FORMAT CODE
    elif "format code" in text or "làm đẹp code" in text or "pho mát cốt" in text:
        return "editor.action.formatDocument"
        
    # 8. ĐỔI GIAO DIỆN SÁNG/TỐI
    elif "chế độ tối" in text or "dark mode" in text or "đác mốt" in text or "đác" in text or "mốt" in text:
        return "workbench.action.toggleLightDarkThemes"

    # Nếu không khớp lệnh nào
    else:
        return "UNKNOWN_COMMAND"

# ==========================================
# 3. API NHẬN AUDIO & XỬ LÝ
# ==========================================
@app.post("/api/voice-command")
async def process_voice_command(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # TẦNG 1: TẠO BỘ TỪ ĐIỂN ÉP BUỘC AI PHẢI NGHE THEO
        tu_vung_goi_y = "tạo file mới, mở terminal, bật terminal, chạy code, format code, chế độ tối, lưu file, đóng tab, xóa terminal."
        
        # ĐÃ SỬA LỖI: Truyền đủ 4 tham số khóa chống ảo giác vào model
        result = model.transcribe(
            temp_file, 
            language="vi", 
            initial_prompt=tu_vung_goi_y,      # Ép AI quét qua từ điển trước khi đoán
            temperature=0.0,                   # Triệt tiêu tính "sáng tạo", lấy kết quả chính xác toán học nhất
            condition_on_previous_text=False,  # Tránh lỗi bịa chuyện nối tiếp
            no_speech_threshold=0.6            # Bỏ qua tạp âm
        )
        
        recognized_text = result["text"]

        print(f"PhoWhisper nghe được: {recognized_text}")

        # Đưa text qua hàm kiểm tra thủ công
        command_id = get_intent_from_text(recognized_text)
        
        return {
            "status": "success",
            "recognized_text": recognized_text,
            "command_id": command_id
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
        
    finally:
        # Xóa file audio rác
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)