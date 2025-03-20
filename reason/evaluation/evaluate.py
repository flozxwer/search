from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from config.config_utils import str2bool
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller
from reason.inference.rm_call import (
    RMRemoteCaller,
    DummyRewardModelCaller,
    RewardModelBaseConfig,
    RemoteRewardModelConfig,
)
from reason.evaluation.evaluator import Task, RemoteMathEvaluator
import torch
from functools import partial
import json
import jsonlines
import time
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import os
import random
from multiprocessing import Pool
import tree
from ray.util.actor_pool import ActorPool
from reason.evaluation.methods import *
import ray

# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)                    # CPU
    torch.cuda.manual_seed_all(seed)           # GPU
    np.random.seed(seed)                       # NumPy
    random.seed(seed)                          # 伪随机数
    os.environ["PYTHONHASHSEED"] = str(seed)   # 字典、集合等数据结构的哈希值
    torch.backends.cudnn.deterministic = True  # cuDNN使用确定性算法

if __name__ == "__main__":
    parser = ArgumentParser()  # 解析器对象（处理命令行、脚本参数）
    parser.add_argument("--LM", type=str, required=True)            # 语言模型
    parser.add_argument("--RM", type=str, default="dummy")          # 奖励模型 默认为dummy
    parser.add_argument("--controller_addr", type=str, default="http://0.0.0.0:28778") # 控制器的地址
    # task config
    parser.add_argument("--task_name", type=str, default="gsm8k")   # 任务名称
    parser.add_argument("--test", type=str2bool, default=True)      # 测试模式（因为命令行输入本质上是字符串，所以使用自定义函数str2bool）
    parser.add_argument("--is_few_shot", type=str2bool, default=False)  # 少量示例进行训练或推理
    parser.add_argument("--seed", type=int, default=0)              # 随机种子（控制实验的可重复性）
    # method config
    parser.add_argument("--method", type=str, required=True)        # 搜索方法
    parser.add_argument("--num_sequence", type=int, default=1)      # 生成样本的数量
    # LM gen config
    parser.add_argument("--temperature", type=float, default=0.0)   # 生成文本的随机性（0：最确定）
    parser.add_argument("--top_k", type=int, default=-1)            # 最可能的 k 个词的数量（-1：没有限制）
    parser.add_argument("--top_p", type=float, default=1)           # 累积概率的阈值
    parser.add_argument("--max_new_tokens", type=int, default=256)  # 生成的最大token数
    # Tree construction config
    parser.add_argument("--tree_max_depth", type=int, default=None)
    parser.add_argument("--tree_max_width", type=int, default=None)
    # ckpg config
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)     # 恢复模型的目录（checkpoint）
    # parallel config
    parser.add_argument("--num_worker", type=int, default=32)       # 工作线程的数量
    parser.add_argument("--local", action="store_true", default=False) # 单线程调试

    config = parser.parse_args() # 解析命令行输入的参数
    setup_seed(config.seed) # 设置实验的随机种子

    # 本地调试
    if config.local:
        print("run in pure local mode for debug only")
        config.num_worker = 1
        ray.init(local_mode=True)

    # TODO(ziyu): move into some configuration file
    # 根据奖励模型的名称（config.RM）来选择不同的步骤标签（step_tag）
    if "math-shepherd" in config.RM.lower(): # lower:小写字母
        prm_step_tag = "ки\n"
    else:
        # assume qwen
        prm_step_tag = "\n\n\n\n\n "
    prm_format_str = "{question} {answer}"

    # 根据语言模型的名称（config.LM）来选择不同的步骤标签
    if "qwen" in config.LM.lower():
        lm_step_tag = "\n\n"
    else:
        lm_step_tag = "ки\n"

    # 创建语言模型生成器
    llm_gen_fn = VLLMRemoteCaller(
        config.LM, config.controller_addr, lm_step_tag=lm_step_tag
    )
    # 创建奖励模型调用器
    if config.RM == "dummy":  # 虚拟
        rm_config = RewardModelBaseConfig(
            step_tag=prm_step_tag, format_str=prm_format_str
        )
        rm_call = DummyRewardModelCaller(rm_config)
    else:
        rm_config = RemoteRewardModelConfig(
            step_tag=prm_step_tag,
            format_str=prm_format_str,
            model_name=config.RM,
            controller_addr=config.controller_addr,
        )
        rm_call = RMRemoteCaller(rm_config)

    # 创建任务对象
    task = Task(task_name=config.task_name, is_few_shot=config.is_few_shot)

    # 并行评估测试集
    def parallel_evaluate_test_dataset(
        method_name: str, solver_fn: Callable, save_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        # 初始化记录器 
        if save_dir is not None:
            record_writer = jsonlines.open(save_dir / f"record.jsonl", mode="w") # 存储评估结果
        else:
            record_writer = None

        test_ds = task.test_ds # 获取测试数据集
        # test_ds = [test_ds[i] for i in range(32)]

        results = []
        # 恢复已有结果
        if config.resume_dir is not None:
            answered_questions = set()
            # 读取已评估结果，添加到 results 列表中
            with jsonlines.open(
                Path(config.resume_dir) / "record.jsonl", "r"
            ) as reader:
                cnt = 0
                for obj in reader:
                    results.append(obj["result"])
                    answered_questions.add(obj["question"]) # 回答过的问题
                    if record_writer is not None:
                        record_writer.write(obj)
                        cnt += 1 # 回答过的问题数
            print(f"Resumed {cnt} questions from {config.resume_dir}")
            total_cnt = len(test_ds)
            # 从 test_ds 中剔除那些已经评估过的问题
            test_ds = [
                problem_inst
                for problem_inst in test_ds
                if problem_inst["question"] not in answered_questions
            ]
            new_cnt = len(test_ds) # 未回答的问题数
            print(
                f"After resuming, there are {new_cnt}/{total_cnt} new questions to answer."
            )

        # 并行计算池
        actor_pool = ActorPool(
            [
                RemoteMathEvaluator.remote(config.task_name, llm_gen_fn, rm_call)
                for _ in range(config.num_worker)
            ]
        )
        # 分配问题实例给并行池中的工作线程进行评估（？）
        res_q = actor_pool.map_unordered(
            lambda p, x: p.evaluate_problem.remote(x, solver_fn), test_ds
        )       # Distributes tasks from the test_ds dataset across the worker pool asynchronously and
                # collects results in any order as they complete. Every worker has a new searching tree as we reset the
                # tree in solver_fn
        # 处理评估结果
        for i, (problem_inst, result, output) in enumerate(
            tqdm(res_q, total=len(test_ds)) # 显示进度条
        ):
            results.append(result)
            if record_writer:
                obj = {
                    # "i": i,
                    "question": problem_inst["question"],
                    "groundtruth": problem_inst["answer"],
                    "result": result,
                    "output": output,
                }
                record_writer.write(obj) # 写入record.jsonl
        # 所有评估结果的平均值
        avg_res = (tree.map_structure(lambda *xs: np.mean(xs), *results),)
        if record_writer:
            json.dump(avg_res, open(save_dir / "avg_result.json", "w"))
        print("Method: {}. Average result: {}".format(method_name, avg_res))
        return results

    # 通过键来访问对应的函数 
    solver_fns = {"cot": cot, "best_of_n": best_of_n}

    # 记录生成过程的配置信息
    cfg_dict_record = dict()
    # XXX: qwen-2.5 requires add more stop words
    # not do it now.
    # stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    # 设置生成任务的参数
    gen_config = LMCallingConfig(
        n=config.num_sequence,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
    )
    cfg_dict_record["gen_config"] = gen_config.__dict__

    # 不同方法对应不同配置
    if config.method == "cot":
        method_config = CoTConfig(config.task_name)
        solver_fn = partial(cot, method_config, gen_config)
    elif config.method == "best_of_n":
        method_config = BestOfNConfig(
            config.task_name, num_sequence=config.num_sequence
        )
        solver_fn = partial(best_of_n, method_config, gen_config)
    elif config.method == "beam_search":
        method_config = BeamSearchConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            beam_size=config.num_sequence,
        )
        solver_fn = partial(beam_search, method_config, gen_config)
    elif config.method == "vanila_mcts":
        method_config = VanilaMCTSConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            select_by_prior=False,
            num_path=config.num_sequence,
        )
        solver_fn = partial(vanila_mcts, method_config, gen_config)
    elif config.method == "rstar_mcts":
        method_config = VanilaMCTSConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            select_by_prior=False,
            num_path=config.num_sequence,
        )
        solver_fn = partial(rstar_mcts, method_config, gen_config)

    else:
        raise ValueError(f"Unknown method: {config.method}")
    cfg_dict_record["method"] = config.method
    cfg_dict_record["method_config"] = method_config.__dict__

    # 记录
    if config.save_dir is not None:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(config.save_dir) / task.task_name / config.method / datetime_str
        save_dir.mkdir(parents=True)
        record_writer = jsonlines.open(save_dir / f"record.jsonl", mode="w")
        cfg_dict_record["LM"] = config.LM
        cfg_dict_record["RM"] = config.RM
        json.dump(cfg_dict_record, open(save_dir / "config.json", "w"))
    else:
        save_dir = None

    parallel_evaluate_test_dataset(config.method, solver_fn, save_dir)
