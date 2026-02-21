from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

model_name = "team-lucid/trocr-small-korean"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

image = Image.open("food1.png").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values

output_ids = model.generate(pixel_values)
print(processor.decode(output_ids[0], skip_special_tokens=True))
