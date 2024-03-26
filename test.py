import cv2
import llama
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "/barn6/yilinsong/LLaMA-Adapter/checkpoints/"

# choose from BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21
model, preprocess = llama.load("BIAS-7B", llama_dir, llama_type="7B", device=device)
model.eval()

prompt = llama.format_prompt("Please introduce this painting.")
img = Image.fromarray(cv2.imread("/barn5/yilinsong/llm/save/JIKA_301_20230309/20230309_050835_1678338575/F/00177.jpg"))
img = preprocess(img).unsqueeze(0).to(device)

result = model.generate(img, [prompt])[0]

print(result)
