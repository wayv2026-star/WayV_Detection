
import os
import shutil
import random

TRAIN_REAL = 'C:/dataset/train/WayV_Detection_Train/train/0_real'
TRAIN_FAKE = 'C:/dataset/train/WayV_Detection_Train/train/1_fake'
VAL_REAL   = 'C:/dataset/val/0_real'
VAL_FAKE   = 'C:/dataset/val/1_fake'

random.seed(42)
VAL_RATIO = 0.1

os.makedirs(VAL_REAL, exist_ok=True)
os.makedirs(VAL_FAKE, exist_ok=True)

def move_to_val(src_dir, dst_dir):
    all_images = []
    for r, d, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(r, f))
    
    val_images = random.sample(all_images, int(len(all_images) * VAL_RATIO))
    
    for img_path in val_images:
        dst_path = os.path.join(dst_dir, os.path.basename(img_path))
        shutil.move(img_path, dst_path)
    
    print(f"{src_dir}: {len(val_images)}장 → val 이동 완료")

move_to_val(TRAIN_REAL, VAL_REAL)
move_to_val(TRAIN_FAKE, VAL_FAKE)
print("완료!")