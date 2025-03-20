#!/bin/sh

CUDA_VISIBLE_DEVICES=7 python -u train_math.py --seed 10 \
                        --dataset_name "prealgebra" \
                        --dataset_path "../envs/math/data/math_500.jsonl" \
                        --model_name_or_path "/mnt/data101_d2/wangzhu/llm_models/Qwen2.5-Math-1.5B-Instruct" \
                        --prm_type "MS" \
                        --prm_model_name_or_path "/mnt/data101_d2/wangzhu/llm_models/peiyi9979-math-shepherd-mistral-7b-prm" \
                        --prm_checkpoint_path "/mnt/data101_d2/wangzhu/checkpoint" \
                        --algorithm_name "TPPO" \
                        --experiment_name "ms_single1.5" \
                        --num_mini_batch 8 \
                        --ppo_epoch 1

