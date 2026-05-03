@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd DFFreq-main
python train.py ^
    --name wayv_progan_detection ^
    --dataroot C:/dataset/WayV_Detection_Train ^
    --train_split train ^
    --val_split val ^
    --batch_size 32 ^
    --niter 90 ^
    --lr 0.0002 ^
    --num_threads 4 ^
    --gpu_ids 0
pause