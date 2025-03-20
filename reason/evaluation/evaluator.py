from dataclasses import dataclass
from datetime import datetime
import importlib
from multiprocessing import Pool
from typing import Any, Callable, Dict, Optional, List, Union

import numpy as np
import ray
from envs import get_default_query_str_builder, get_env_datasets
from reason.inference.lm_call import LanguageModelCallingFunction
from reason.inference.rm_call import RewardModelCallingFunction
from reason.reranking.vote_utils import (
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_VOTE,
    PRM_LAST_MAX,
    AGG_FN_MAP,
)
from envs.base_env import INVALID_ANS

class Task:
    def __init__(self, task_name: str, is_few_shot: bool = False):
        """
        初始化 Task 类，加载任务模块并根据任务名称初始化相关属性。
        
        :param task_name: 任务名称，用于加载对应的任务模块
        :param is_few_shot: 是否为少样本任务，默认为 False
        """
        self.task_name = task_name
        # 动态导入任务模块(在项目envs目录下)
        task_module = importlib.import_module(f"envs.{task_name}")
        # 根据任务名称初始化相关的函数
        if task_name == "MATH" or task_name == "rstar":
            # 如果是 MATH 或 rstar 任务，加载相应的处理函数
            self.extract_answer = task_module.extract_answer
            self.extract_groundtruth = task_module.extract_groundtruth
            self.judge_correct = task_module.judge_correct
        else:
            # 对于不支持的任务，抛出异常
            raise NotImplementedError(f"Task {task_name} is not supported")

        # 设置是否为少样本任务
        self._is_few_shot = is_few_shot
        
        # 获取对应任务的环境函数
        self.env_fn = task_module.Env

    def prompt_fn(self, problem_input: str):
        """
        根据问题输入构建对应的提示字符串。
        
        :param problem_input: 输入的任务问题
        :return: 构建好的提示字符串
        """
        return get_default_query_str_builder(self.task_name)(
            problem_input, is_few_shot=self._is_few_shot
        )

    @property
    def test_ds(self):
        """
        获取任务的测试数据集。
        
        :return: 任务的测试数据集
        """
        return get_env_datasets(self.task_name)[1]


CHOSEN_AGGR_METHODS = [
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_MAX,
    PRM_LAST_VOTE,
]


def judge_ans(
    problem_str: str,
    extracted_groundtruth: str,
    output_list: List[str],
    v_list: List[float],
    aggration_mode: str,
    extract_answer_fn,
    judge_correct_fn,
    normalize=False,
):
    # 提取有效答案和相应的置信度值
    ans_list = [extract_answer_fn(txt) for txt in output_list]
    valid_ans_list, valid_v_list = [], []
    for i, ans in enumerate(ans_list):
        if ans != INVALID_ANS:
            valid_ans_list.append(ans)
            valid_v_list.append(v_list[i])
    if len(valid_ans_list) == 0:
        return 0

    # 归一化置信度值
    if "orm" in aggration_mode and normalize:
        # score_normalization: this is only necessary for [-1, 1] values
        valid_v_list = np.array(valid_v_list)
        valid_v_list -= valid_v_list.min()
        valid_v_list /= valid_v_list.max() + 1e-3
        valid_v_list = valid_v_list.tolist()
    # 聚合答案
    aggregated_ans = AGG_FN_MAP[aggration_mode](valid_ans_list, valid_v_list)

    # 判断聚合答案是否正确
    return (
        1 if judge_correct_fn(problem_str, extracted_groundtruth, aggregated_ans) else 0
    )


@dataclass
class SolutionOutput:
    solutions: List[str]
    # Define the completion tokens for each solution
    #  For best_of_n, it's a list of int, indicate how many tokens in each
    #      generation
    #  for beam search, it's a list of zeros, except the last element indicates total tokens
    #  for mcts, it's a list of int, indicate how many tokens comsumed between two paths
    # 定义每个解决方案的完成标记
    # 对于best_of_n，它是一个整数列表，表示每一代中有多少标记
    # 对于束搜索，它是一个零列表，除了最后一个元素表示总标记数
    # 对于mcts，它是一个整数列表，表示两条路径之间消耗了多少标记
    completion_tokens: List[int]


@dataclass
class TreeSearchSolutionOutput(SolutionOutput):
    tree_completion_tokens: List[int]


class MathEvaluator:

    def __init__(
        self,
        task: Union[str, Task],
        lm_call: LanguageModelCallingFunction,
        rm_call: RewardModelCallingFunction,
    ):
        # 初始化任务对象，如果传入的是字符串，则转换为 Task 对象
        if isinstance(task, str):
            self._task = Task(task_name=task)
        else:
            assert isinstance(task, Task)
            self._task = task
        # 保存语言模型调用和奖励模型调用的函数
        self.lm_call = lm_call
        self.rm_call = rm_call

    def evaluate_problem(
        self, problem_inst: Dict[str, str], solver_fn: Callable
    ) -> List[str]:
        # 通过 solver_fn 求解问题实例，获得答案和其他信息
        solution: SolutionOutput = solver_fn(problem_inst, self.lm_call, self.rm_call)
        # 分析输出结果，并得到评估结果和详细的输出列表
        result, output = self.analyze_output(problem_inst, solution.solutions)

        # 计算总的完成令牌数
        total_completion_token = 0
        for i, o in enumerate(output):
            # 为每个输出项附加对应的完成令牌数
            o["completion_tokens"] = solution.completion_tokens[i]
            if isinstance(solution, TreeSearchSolutionOutput):
                # 如果是树搜索输出，还需要附加树搜索的令牌数
                o["tree_completion_tokens"] = solution.tree_completion_tokens[i]
            # We define the completion_tokens as the tokens comsumed between two generated
            #  answers, therefore we need to take sum here.
             # 计算所有生成的答案的总令牌数
            total_completion_token += solution.completion_tokens[i]

        # 将总令牌数加入到结果中
        result["total_completion_tokens"] = total_completion_token
        return problem_inst, result, output

    def analyze_output(self, problem_inst: Dict[str, str], gen_answers: List[str]):
        # 从问题实例中提取答案的真实值
        extracted_groundtruth = self._task.extract_groundtruth(problem_inst["answer"])

        # 如果有多个生成的答案
        if len(gen_answers) > 1:
            # 生成的每个答案和问题都组合成一个输入对
            input_list = [(problem_inst["question"], txt) for txt in gen_answers]
            # XXX(ziyu): for tree search methods with value_fn, should not call rm 
            #  to compute it again
            # 通过奖励模型计算每个答案的值
            value_list = self.rm_call(input_list, lm_step_tag=self.lm_call.lm_step_tag)
        else:
            # 如果只有一个答案，则默认值为 0
            value_list = [[0]]
        # 将生成的答案和对应的值组合成输出列表
        output_list = [
            {"path_idx": i, "text": txt, "value": v}
            for i, (txt, v) in enumerate(zip(gen_answers, value_list))
        ]

        # 根据选择的聚合方法，判断每个答案的正确性
        res = {
            agg_method: judge_ans(
                problem_inst["question"],             # 问题文本
                extracted_groundtruth,      # 真实答案
                gen_answers,                          # 生成的答案列表
                value_list,                                # 生成答案对应的值
                agg_method,                        # 聚合方法
                self._task.extract_answer,      # 提取答案的函数
                self._task.judge_correct,        # 判断答案是否正确的函数
            )
            for agg_method in (
                CHOSEN_AGGR_METHODS if len(gen_answers) > 1 else [MAJORITY_VOTE]
            )
        }
        return res, output_list


@ray.remote # 在 Ray 集群上远程执行的类
class RemoteMathEvaluator(MathEvaluator):
    def __init__(
        self,
        task: str,
        lm_call: LanguageModelCallingFunction,
        rm_call: RewardModelCallingFunction,
    ):
        # 调用父类的构造函数，初始化任务、语言模型调用和奖励模型调用
        super().__init__(task, lm_call, rm_call)
