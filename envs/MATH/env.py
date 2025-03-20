import copy
import re
from typing import List, Optional
import numpy as np
from envs.base_env import CoTEnv, NoLegalActionException, INVALID_ANS
from .prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP
# from .verify_utils import extract_answer as extract_fn, grade_answer
from .parse_utils_qwen import extract_answer as extract_fn, parse_ground_truth
from .grader import math_equal

ANS_RE = None 
STOP_STR = None

# 提取答案的函数
def extract_answer(answer_str: str) -> str:
    # 调用 extract_fn 函数提取答案，数据名称为 'math'
    return extract_fn(answer_str, data_name='math')

# 提取 Groundtruth (真实答案) 的函数
def extract_groundtruth(groundtruth_str: str) -> str:
    # 调用 parse_ground_truth 函数提取真实答案，固定数据名称为 'math'
    return parse_ground_truth(groundtruth_str, data_name='math')

# 判断答案是否正确的函数
def judge_correct(
    problem_str: str, extracted_groundtruth: Optional[str], answer: str
) -> bool:
    # return grade_answer(given_answer=answer, ground_truth=extracted_groundtruth)
    # 使用 math_equal 函数比较提取的答案和实际答案是否相等
    result = math_equal(answer, extracted_groundtruth)
    return result


# Env 类继承自 CoTEnv，定义了一个包含数学问题环境的类
class Env(CoTEnv):
    sep = SEP  # 定义一个分隔符常量（SEP）

    # 初始化函数
    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn,
        task_desc_str: str = COT_TASK_DESC,
        cot_example_str: str = COT_EXAMPLES,
        problem_format_str: str = PROBLEM_FORMAT_STR,
        reset=True,
    ):
        # 调用父类 CoTEnv 的初始化函数，初始化环境
        super().__init__(
            config,
            math_problems,
            llm_gen_fn,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset,
        )

    # 定义一个停止字符串的属性，用于中止操作
    @property
    def stop_str(self):
        return STOP_STR

    # 后处理行动的函数，确保行动字符串以分隔符结尾
    def post_process_act(self, action: str):
        # 如果行动字符串没有以分隔符结尾，添加分隔符
        if not action.endswith(self.sep):
            action = action.strip() + self.sep
        
        return action

    # 判断一个动作是否正确的函数
    def _is_correct(self, completion):
        # 提取动作中的答案
        extracted_answer = extract_answer(completion)
        # print("Compare: {} -- {}".format(extrated_answer,
        #  self.math_problem['answer']))
        # return extrated_answer == self.math_problem['answer']
        # 调用 judge_correct 函数来判断提取的答案与实际答案是否一致
        return judge_correct(
            self.math_problem["question"], self.math_problem["answer"], extracted_answer
        )

    # 获取奖励的函数（该函数未实现，需要根据学到的奖励模型来实现）
    def get_reward(self):
        """To implement based on learned reward model"""
        return 0  # 目前没有实现奖励计算，返回 0
