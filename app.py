from transformers import pipeline
from PIL import Image
import torch

# GPU 사용 여부 설정
device = 0 if torch.cuda.is_available() else -1

# DeepSeek-OCR 파이프라인 생성
pipe = pipeline(
    "image-text-to-text",
    model="deepseek-ai/DeepSeek-OCR",
    trust_remote_code=True,
    device=device
)

# 이미지 로드
image_path = "sample.png"   # <-- 여기에 테스트 이미지 경로 입력
image = Image.open(image_path).convert("RGB")

# OCR 실행
result = pipe(image)

# 결과 출력
text = result[0]["generated_text"]
print(text)