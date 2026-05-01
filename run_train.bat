@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd DFFreq-main
python train.py ^
    --name wayv_face_detection ^
    --dataroot C:/dataset ^
    --train_split train/WayV_Detection_Train/train ^
    --val_split val ^
    --batch_size 32 ^
    --niter 90 ^
    --lr 0.0002 ^
    --num_threads 4 ^
    --gpu_ids 0
pause