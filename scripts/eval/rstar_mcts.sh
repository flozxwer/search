python reason/evaluation/evaluate.py \
    --LM Qwen2.5-Math-7B-Instruct \
    --RM peiyi9979-math-shepherd-mistral-7b-prm \
    --task_name rstar \
    --max_new_tokens 2048 \
    --num_sequence 1 \
    --tree_max_width 4 \
    --tree_max_depth 50 \
    --save_dir /mnt/data101_d2/wangzhu/output/log/inference \
    --method rstar_mcts \
    --num_worker 32 \
    --controller_addr http://0.0.0.0:28777 \
    --top_k 40 \
    --top_p 0.95 \
    --temperature 0.8 \
#    --local