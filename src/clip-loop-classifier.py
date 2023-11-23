import torch
import clip
from PIL import Image
import os
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

data_path = "/app/smoke-detect/tracking/src/old-code/output/1/"

# List all files in the folder
files = os.listdir(data_path)

inx = 0

# Loop through each file and read its contents
for file_name in files:
    file_path = os.path.join(data_path, file_name)

    image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
    text = clip.tokenize(["no smoke vehicle", "smoke vehicle"]).to(device)
    
    inx += 1

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
    
    img = cv2.imread(file_path)
    
    # Find the index of the maximum value
    max_index = np.argmax(probs)

    # Get the maximum value itself
    # max_value = probs[max_index]

    # print("Max Value:", max_value)
    print("Index of Max Value:", max_index)
    
    
    target = ["no smoke vehicle", "smoke vehicle"]
    
    print(f"Prediction : {target[max_index]}")
    
    print("------------------------------------")
    
    cv2.imwrite(f"/app/smoke-detect/clip-classifier/output/{max_index}/class_{target[max_index]}_{inx}.jpg", img)