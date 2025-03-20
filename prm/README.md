#  Supervised Training for PRMs

cd prm/code

\\ single gpu
python finetune_qwen_single_gpu.py --model_path $YOUR_MODEL_PATH \
                                   --train_data_path $TRAIN_DATA_PATH \
                                   --test_data_path $TEST_DATA_PATH


\\ multi gpu
torchrun --nproc_per_node=2 finetune_qwen.py --model_path $YOUR_MODEL_PATH \
                                             --data_path $YOUR_DATA_FOLDER_PATH \
                                             --datasets both \