import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("image.png").convert('RGB')
txt = "Gambar"
inputs = processor(images=image, text=txt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=30)

caption = processor.decode(outputs[0],skip_special_tokens=True)
print(caption)