@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd DFFreq-main
python test.py ^
    --model_path ./checkpoints/wayv_progan_detection2026_05_03_00_07_13/model_epoch_last.pth ^
    --num_threads 0 ^
    --gpu_ids 0
pause