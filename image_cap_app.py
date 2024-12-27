import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

def caption_img(input_image:np.ndarray):
    raw_img = Image.fromarray(input_image).convert('RGB')
    inputs = processor(images=raw_img, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=30)
    caption = processor.decode(outputs[0],skip_special_tokens=True)
    return caption
    
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

iface = gr.Interface(
    fn=caption_img,
    inputs=gr.Image(),
    outputs="text",
    title="Image Captioning",
    description="Simple Web App to Demonstrate Image Captioning Using BLIP"
)

iface.launch()