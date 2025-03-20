from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from reason.inference.value import _value_inference_fastchat


@dataclass
class RewardModelBaseConfig:
    step_tag: str # 步骤标签
    # a format string that takes in question and answer
    #  need to have {question} and {answer} in the string
    format_str: str # 格式化字符串


class RewardModelCallingFunction:
    # 初始化，接收配置对象并存储必要的配置信息
    def __init__(self, config: RewardModelBaseConfig):
        self.config = config  # 存储配置对象
        self.step_tag = config.step_tag  # 步骤标签
        self.format_str = config.format_str  # 格式字符串

    # 该方法未实现，应该在子类中实现，根据问题-答案对评估奖励值
    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        lm_step_tag: str,
    ) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError

    # 替换答案中的步骤标签（lm_step_tag），以使用当前配置中的步骤标签（step_tag）
    def replace_step_tag(self, answer: str, lm_step_tag: str):
        # 根据lm_step_tag分割答案
        splits = answer.split(lm_step_tag)
        # 清理每个分割后的部分，去除首尾空白字符
        splits = [s.strip() for s in splits]
        # 用新的step_tag连接分割的部分，并避免tokenization问题（添加空格）
        response = f" {self.step_tag}".join([s for s in splits if s != ""])
        # 在答案末尾添加step_tag
        response += f" {self.step_tag}"
        return response



class DummyRewardModelCaller(RewardModelCallingFunction):
    # a dummy rm caller that always return 0
    # 一个虚拟的奖励模型调用器，总是返回 0

    def __init__(self, config: RewardModelBaseConfig):
        super().__init__(config) # 调用父类的构造函数初始化配置

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        lm_step_tag: str,
    ) -> Union[List[int], List[List[int]]]:
        # 辅助函数，用于生成步骤编号
        def fn(s):
            steps = s.split(self.step_tag)                  # 按照步骤标签分割字符串
            steps = [s for s in steps if s.strip() != ""]   # 去除空的步骤
            return list(range(len(steps)))                  # 返回步骤编号的列表

        if isinstance(question_answer_pairs[0], str):
            # 如果是单个问答对，格式化问题和答案后调用辅助函数
            return fn(
                self.format_str.format(
                    question=question_answer_pairs[0],
                    answer=self.replace_step_tag(question_answer_pairs[1], lm_step_tag),
                )
            )
        else:
            # 如果是多个问答对，遍历每个问答对并处理
            return [
                fn(
                    self.format_str.format(
                        question=s[0],
                        answer=self.replace_step_tag(s[1], lm_step_tag),
                    )
                )
                for s in question_answer_pairs
            ]


@dataclass
class RemoteRewardModelConfig(RewardModelBaseConfig):
    model_name: str
    controller_addr: str


class RMRemoteCaller(RewardModelCallingFunction):
    def __init__(self, config: RemoteRewardModelConfig):
        # 初始化，设置模型名称和控制器地址，并调用父类（RewardModelCallingFunction）初始化方法
        self.model_name = config.model_name
        self.controller_addr = config.controller_addr
        super().__init__(config)

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        lm_step_tag: str,
    ) -> Union[List[int], List[List[int]]]:
        """
        处理问题和答案对，并调用远程模型进行推理。

        - 如果问题和答案是单个对（tuple），格式化后进行推理。
        - 如果有多个问题和答案对（list），格式化后批量进行推理。

        参数：
        - question_answer_pairs: 问题和答案的对，可能是单个 tuple 或者多个 tuple 的列表
        - lm_step_tag: 步骤标记，用于替换答案中的特定部分

        返回：
        - 返回模型推理结果，通常是整数列表（对应模型评分等）
        """

        if isinstance(question_answer_pairs[0], str):
            # 如果传入的是单个问题和答案对，进行格式化处理
            response = self.replace_step_tag(question_answer_pairs[1], lm_step_tag)
            input_str = self.format_str.format(
                question=question_answer_pairs[0], answer=response
            )
        else:
            # 如果传入的是多个问题和答案对，批量处理格式化
            input_str = [
                self.format_str.format(
                    question=s[0],
                    answer=self.replace_step_tag(s[1], lm_step_tag),
                )
                for s in question_answer_pairs
            ]
        # 调用外部推理服务，传递格式化后的输入字符串
        return _value_inference_fastchat(
            input_str=input_str,
            model_name=self.model_name,
            controller_addr=self.controller_addr,
        )
