import torch
import sys
import io
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from torchvision import transforms
from PIL import Image

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'DFFreq-main'))

from networks.resnet import resnet50
from app.gradcam import GradCAM, apply_heatmap

app = FastAPI()

# 모델 로드
MODEL_PATH = os.path.join(BASE_DIR, 'DFFreq-main', 'checkpoints', 'wayv_progan_detection2026_05_03_00_07_13', 'model_epoch_last.pth')

model = resnet50(num_classes=1)
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict, strict=True)
model.eval()

# Grad-CAM 타겟 레이어 설정
target_layer = model.layer2[-1]
gradcam = GradCAM(model, target_layer)

# 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    if file.content_type not in ['image/png', 'image/jpeg', 'image/jpg']:
        return JSONResponse(
            {"error": "지원하지 않는 파일 형식입니다. PNG, JPG만 가능합니다."},
            status_code=400
        )


    # 이미지 읽기
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # 추론
    input_tensor = transform(image).unsqueeze(0)
    input_tensor.requires_grad_(True)
    
    with torch.enable_grad():
        cam = gradcam.generate(input_tensor)
    
    with torch.no_grad():
        output = model(transform(image).unsqueeze(0))
        fake_prob = torch.sigmoid(output).item()
    
    # 판정
    if fake_prob >= 0.9:
        verdict = "AI 생성 이미지입니다"
        color = "#ff4444"
    elif fake_prob >= 0.7:
        verdict = "AI 생성 이미지로 의심됩니다"
        color = "#ff8800"
    elif fake_prob >= 0.3:
        verdict = "판단이 어렵습니다"
        color = "#ffcc00"
    else:
        verdict = "실제 이미지입니다"
        color = "#44bb44"
    
    # 히트맵 생성
    heatmap_img = apply_heatmap(image, cam)
    
    return JSONResponse({
        "verdict": verdict,
        "color": color,
        "fake_prob": round(fake_prob * 100, 1),
        "original": image_to_base64(image.resize((256, 256))),
        "heatmap": image_to_base64(heatmap_img)
    })

app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "app", "static"), html=True), name="static")
