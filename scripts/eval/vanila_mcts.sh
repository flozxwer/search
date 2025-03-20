python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-7B-Instruct \
    --RM peiyi9979-math-shepherd-mistral-7b-prm \
    --controller_addr http://0.0.0.0:28777 \
    --task_name MATH \
    --method vanila_mcts \
    --num_sequence 1 \
    --temperature 0.7 \
    --max_new_tokens 2048 \
    --tree_max_depth 50 \
    --tree_max_width 4 \
    --save_dir /mnt/data101_d2/wangzhu/output/log/inference \
    --num_worker 32 \
    --local

# math-shepherd-mistral-7b-prm