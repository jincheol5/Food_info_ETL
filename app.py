import cv2
import easyocr



# ----------------------------
# 3. EasyOCR 적용
# ----------------------------

reader = easyocr.Reader(['ko','en'], gpu=False)

results = reader.readtext("processed.png")

# ----------------------------
# 4. 텍스트만 출력
# ----------------------------

for bbox, text, confidence in results:
    print(text)
