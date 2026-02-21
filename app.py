import os
import torch
from transformers import AutoModel, AutoTokenizer

# =========================
# 환경 설정
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = "deepseek-ai/DeepSeek-OCR"

# =========================
# 모델 & 토크나이저 로드
# =========================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

print("Loading model...")
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True
    # _attn_implementation="flash_attention_2" ← 제거
)

model = model.eval().cuda().to(torch.bfloat16)

print("Model loaded successfully.")

# =========================
# 입력 설정
# =========================
prompt = "<image>\nFree OCR. "

image_file = "food1.png"  # ← 실제 이미지 경로로 수정
output_path = "./ocr_results"

# 이미지 존재 확인
if not os.path.exists(image_file):
    raise FileNotFoundError(f"Image not found: {image_file}")

# 출력 폴더 생성
os.makedirs(output_path, exist_ok=True)

# =========================
# 추론 실행
# =========================
with torch.no_grad():
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        output_path=output_path,
        base_size=1024,   # Base 모델 권장
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=True
    )

print("OCR Finished.")
print("Result:\n", result)