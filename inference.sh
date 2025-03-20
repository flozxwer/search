export PYTHONPATH=$(pwd)
sh scripts/eval/cot_greedy.sh
# 5min
# Method: cot. Average result:[{"majority_vote": 0.836, "total_completion_tokens": 630.52}]

sh scripts/eval/cot_rerank.sh
# 1h
# Method: best_of_n. Average result:[{"majority_vote": 0.864, "prm_min_max": 0.852, "prm_min_vote": 0.874, "prm_last_max": 0.854, "prm_last_vote": 0.874, "total_completion_tokens": 5286.584}]

sh scripts/eval/beam_search.sh
# 2h
# Method: beam_search. Average result: [{"majority_vote": 0.852, "total_completion_tokens": 2832.62}]

sh scripts/eval/vanila_mcts.sh
# 7h
# Method: MCTS. Average result:[{"majority_vote": 0.818, "total_completion_tokens": 2979.92}]

#sh scripts/eval/rstar_mcts.sh
# 2h
