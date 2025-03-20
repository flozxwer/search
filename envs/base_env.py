import abc
from typing import Dict, List, Optional
import numpy as np
import copy
import pdb
import torch
from distributed.utils import print_with_rank
from transformers import PreTrainedTokenizer
from reason.inference.lm_call import LMCallingConfig, ConcatedLMGenResult

INVALID_ANS = "[invalid]"


class NoLegalActionException(Exception):
    pass


class ResetException(Exception):
    pass


class BaseEnv(abc.ABC):
    """Basic environment to use for MCTS"""
    """用于MCTS（蒙特卡洛树搜索）的基本环境类"""

    @abc.abstractmethod
    def reset(self, update_legal_action: bool):
        """重置环境，通常用于开始新的游戏或任务。

        Args:
            update_legal_action (bool): 如果为 True，环境将更新合法的动作列表。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self):
        """执行一步操作，通常用于执行一个动作并返回新的环境状态。

        返回值:
            一个包含状态更新、奖励等信息的元组。
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def legal_actions(self):
        """返回当前状态下所有合法的动作。

        返回值:
            一个包含所有合法动作的集合或列表。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self):
        """返回环境的副本，通常用于保存当前状态。

        返回值:
            当前环境的一个副本。
        """
        raise NotImplementedError

    @staticmethod
    def build_query_str(
        cot_task_desc: Optional[str],
        cot_examples: Optional[str],
        problem_format_str: str,
        problem_input: str,
        is_few_shot: bool = False,
    ):
        """a wrap function that wrap the problem text with certrain format
        e.g. prompt_str = "Input: " + join_numbers(" ", xs) + "\nSteps:\n"
        >>> query_str = Game24Env.build_query_str("1 1 1 1")
        >>> print(query_str)
        >>> Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
        Input: 1 1 1 1
        Steps:

        >>>
        """

        """一个包装函数，用于将问题文本与某种格式结合，以创建一个查询字符串。
        
        例如，给定一组输入数字，函数可以生成一个描述如何将这些数字合并成目标值（比如24）的任务描述字符串。

        Args:
            cot_task_desc (Optional[str]): 任务描述文本，用于解释当前问题。
            cot_examples (Optional[str]): 示例，通常是问题和答案对，用于少样本学习。
            problem_format_str (str): 问题格式字符串，可以插入问题输入的地方。
            problem_input (str): 问题的输入部分，通常是输入数据（如一组数字）。
            is_few_shot (bool, optional): 如果为True，表示这是少样本学习任务，函数将包括示例。默认为False。

        返回:
            str: 构建好的查询字符串。
        
        示例：
        >>> query_str = Game24Env.build_query_str("1 1 1 1")
        >>> print(query_str)
        Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
        Input: 1 1 1 1
        Steps:
        >>> 
        """

        ret = ""

        # 如果提供了任务描述（cot_task_desc），则将其添加到查询字符串中
        if cot_task_desc:
            ret += cot_task_desc + "\n"
        # 如果启用了少样本（is_few_shot），则将示例添加到查询字符串中
        if is_few_shot:
            ret += cot_examples + "\n"
        # 将格式化字符串和问题输入结合起来，形成最终的查询字符串
        ret += problem_format_str.format(question=problem_input)

        return ret

    @staticmethod
    def build_response_str(
        answer_str: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool
    ):
        """构建并返回一个响应字符串，通常用于将模型的回答转换为适当的格式。

        Args:
            answer_str (str): 模型生成的答案字符串。
            tokenizer (PreTrainedTokenizer): 用于处理文本的分词器，通常用于将字符串转换为模型能处理的格式。
            add_eos_token (bool): 是否在回答后添加结束符（End Of Sequence，EOS）标记。

        返回:
            str: 格式化后的响应字符串，可能会在结尾加上 EOS 标记。
        """
        raise NotImplementedError


class CoTEnv(BaseEnv):
    """The basic environment for solving natural language problems using CoT"""
    """用于解决自然语言问题的CoT（Chain-of-Thought）基础环境类"""

    sep: str # 定义分隔符，后续可能在问题文本中使用（此处暂未具体化）

    @property
    def stop_str(self):
        """停止标志字符串，当生成的文本遇到该字符串时，应该停止生成"""
        return NotImplementedError

    def _is_correct(self, completion) -> bool:
        """检查给定的完成文本是否正确。

        Args:
            completion (str): 模型生成的答案文本

        返回:
            bool: 如果答案正确，返回True；否则返回False
        """
        raise NotImplementedError

    def get_reward(self):
        """根据学习到的奖励模型计算奖励值

        该方法应根据完成任务的质量（如生成的文本是否正确、是否符合预期等）返回一个奖励值
        """
        """To implement based on learned reward model"""
        raise NotImplementedError

    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn, # lm_call: LanguageModelCallingFunction
        task_desc_str: str,
        cot_example_str: str,
        problem_format_str: str,
        reset=True,
    ):
        """初始化CoTEnv环境

        Args:
            config (dict): 配置信息，包含如是否使用少样本学习等配置
            math_problems (list): 数学问题列表，用于任务
            llm_gen_fn (function): 用于生成自然语言模型输出的函数
            task_desc_str (str): 任务描述字符串
            cot_example_str (str): CoT示例字符串，用于少样本学习
            problem_format_str (str): 问题格式字符串
            reset (bool): 是否初始化时重置环境，默认是True
        """
        self.config = config
        self.mcts_mode = "play_with_bot_mode"               # 设置为“与机器人对战模式”
        self.math_problems = math_problems                  # 数学问题列表
        self.llm_gen_fn = llm_gen_fn                        # LLM生成函数
        self.action_history = None                          # 行动历史
        self.math_problem = None                            # 当前数学问题
        self._legal_actions = None                          # 合法动作列表
        self.is_few_shot = config.get("is_few_shot", False)  # 是否使用少样本学习模式

        # 任务描述、CoT示例和问题格式字符串
        self._task_desc_str = task_desc_str
        self._cot_example_str = cot_example_str
        self._problem_format_str = problem_format_str

         # 构建任务前缀（task_prefix），如果任务描述或示例存在，则将其添加
        prefixes = []
        if self._task_desc_str is not None:
            prefixes.append(self._task_desc_str)
        if self.is_few_shot:
            prefixes.append(self._cot_example_str)
        if len(prefixes) > 0:
            self.task_prefix = "\n".join(prefixes)  # 将任务描述和示例连接成一个字符串
        else:
            self.task_prefix = None

        # 如果reset为True，则初始化环境
        if reset:
            self.reset(update_legal_action=True)

    def reset(self, update_legal_action=True):
        """重置环境，通常用于开始一个新的任务或游戏

        Args:
            update_legal_action (bool): 是否更新合法动作列表，默认为True

        返回:
            state: 重置后的状态
            info: 额外的返回信息，包括API完成标记
        """
        # 设置问题索引为0，通常代表开始的数学问题
        # reset environment to problem idx
        self.set_problem(idx=0)
        self.action_history = []  # 清空行动历史
        # 构建查询字符串，包含任务描述、示例和问题输入
        self._init_query = self.build_query_str(
            cot_examples=self._cot_example_str,
            cot_task_desc=self._task_desc_str,
            problem_format_str=self._problem_format_str,
            problem_input=self.math_problem["question"],
            is_few_shot=self.is_few_shot,
        )
        # 如果需要更新合法动作列表，尝试更新最多3次
        if update_legal_action:
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    # 更新合法动作
                    self._legal_actions, api_completion_token = (
                        self.update_legal_actions()
                    )
                    break
                except NoLegalActionException as e:
                    # 如果没有合法动作，最多重试3次
                    if cnt == 3:
                        raise ResetException
        # 返回当前状态及额外信息（如API完成标记）
        info = {"api_completion_token": api_completion_token}
        return self.get_state(), info

    def step(self, action, update_legal_action=True):
        """执行一步操作，更新环境状态

        Args:
            action (str): 执行的动作
            update_legal_action (bool): 是否更新合法动作列表，默认为True

        返回:
            state: 当前状态
            reward: 当前奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外的信息（如API完成标记、获胜者等）
        """
        self.action_history.append(action)  # 将当前动作添加到行动历史
        state = self.get_state()  # 获取当前状态
        reward = self.get_reward()  # 计算当前奖励
        terminated, truncated, info = self.get_done_and_info()  # 获取是否终止或截断的标志以及额外信息

        # 如果未终止或截断，并且需要更新合法动作列表
        # update legal actions
        if not (terminated or truncated) and update_legal_action:
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    # 更新合法动作并将API完成标记加入信息
                    self._legal_actions, api_completion_token = self.update_legal_actions()
                    info["api_completion_token"] = api_completion_token
                    break
                except NoLegalActionException as e:
                    # 如果没有合法动作，最多重试3次
                    if cnt == 3:
                        terminated = True
                        reward = 0
                        self._legal_actions = None
                        info["winner"] = 2  # 假设如果没有合法动作，游戏的赢家为2
                        info["api_completion_token"] = 0
                    else:
                        pass
        else:
            self._legal_actions = None  # 如果游戏终止或截断，设定合法动作为None
            if info["winner"] == 1:
                reward = 1.0  # 如果玩家1获胜，奖励为1.0
            info["api_completion_token"] = 0  # 设置API完成标记为0
        return state, reward, terminated, truncated, info

    def get_state(self):
        # not join about sep_str here because we let vllm return with sep_str
        # 该方法返回当前状态，状态是由初始化的查询（_init_query）和所有历史动作（action_history）拼接而成
        # 注意：我们不在此处连接sep_str，因为让vllm在返回结果时会处理sep_str
        ret = self._init_query + "".join(self.action_history)
        return ret

    def post_process_act(self, action: str):
        # This step may change the token count
        # 该步骤可能会改变令牌的数量，用来后处理动作，处理后返回动作
        return action

    def update_legal_actions(self):
        # 更新合法动作列表。通过调用llm_gen_fn生成一系列动作，并根据生成的结果进行筛选和处理
        result: ConcatedLMGenResult = self.llm_gen_fn(
            input_str=self.get_state(),  # 当前状态（包含历史动作）
            config=LMCallingConfig(
                n=self.config["max_actions"],  # 最大动作数（树的宽度）
                stop_str=self.sep,  # 停止符
                include_stop_str_in_output=True,  # 在输出中包含停止符
                **self.config["generation_config"]  # 其他生成配置
            ),
        )
        texts = result.text  # 从生成结果中提取文本
        logps_avg_by_len = result.logp_avg_by_len  # 每个生成文本的平均对数概率
        token_len = result.num_tokens  # 每个生成文本的令牌数量
        text_list, prob_list, num_token_list = [], [], []  # 用于存储合法动作、概率、令牌数量
        finish_reason_list = []  # 用于存储每个生成动作的结束原因
        next_state_terminated = {}  # 用于存储每个动作是否终止

        for i in range(len(texts)):
            # XXX: this process can be improve or moved to other place
            # this is a pre-judge of terminal flag or certain action, by
            # whether the text-generation is stop by the <eos> or stop_str
            #  XXX: 此过程可以优化或移到其他地方
            # 这是一个预判步骤，用于判断生成文本是否是终止标志或者某个特定动作，
            # 判断方式是看文本生成是否被<eos>或stop_str终止
            
            terminated = not texts[i].endswith(self.sep) # 如果文本没有以sep结束，说明该动作未被完全生成

            processed_act = self.post_process_act(texts[i]) # 后处理生成的动作
            if (
                len(processed_act) > 0 # 动作不能为空
                and processed_act not in text_list # 动作必须是唯一的
                # only stop is valid, otherwise the output action is truncated actually
                and result.finish_reason[i] == "stop"  # 只有当生成是由于stop标志终止时，才认为它是有效的
            ):
                text_list.append(processed_act)  # 添加合法动作
                prob_list.append(logps_avg_by_len[i])  # 添加该动作的平均概率
                num_token_list.append(token_len[i])  # 添加该动作的令牌数量
                finish_reason_list.append(result.finish_reason[i])  # 添加该动作的结束原因
                next_state_terminated[processed_act] = terminated  # 记录该动作是否终止

        if len(prob_list) == 0:
            # 如果没有生成任何合法动作，则打印状态信息并抛出异常
            print_with_rank("state: {}".format(self.get_state()))
            print_with_rank("gen_result: {}".format(result))
            raise NoLegalActionException("No possible action have been generated.")

        prob_list = np.exp(prob_list)  # 将概率转换为指数形式
        prob_list = np.array(prob_list)  # 转换为numpy数组
        # normalize probability
        # 对概率进行归一化
        prob_list = prob_list / np.sum(prob_list)

        # 生成合法动作列表，每个动作包含动作内容、概率、令牌数和结束原因
        _legal_actions = [
            {
                "action": action,
                "prob": prob,
                "num_token": n_token,
                "finish_reason": finish_reason,
            }
            for action, prob, n_token, finish_reason in zip(
                text_list, prob_list, num_token_list, finish_reason_list
            )
        ]

        # 更新下一个状态是否终止
        self._next_state_terminated = next_state_terminated
        return _legal_actions, result.completion_tokens # 返回合法动作列表和生成的完成令牌数

    def set_problem(self, idx):
        # 设置当前的数学问题，通过索引从math_problems中选取问题
        self.math_problem = self.math_problems[idx]

    @property
    def query(self):
        # 返回初始化的查询内容
        return self._init_query

    @property
    def question(self)->str:
        # 返回当前数学问题的题目
        return self.math_problem["question"]

    @property
    def answer(self):
        # 返回由历史动作生成的答案
        return "".join(self.action_history)

    def get_done_and_info(self):
        # 获取是否完成的问题及相关信息。判断是否达到最大长度或生成停止符时认为完成
        info = {"winner": 0}  # 默认没有赢家
        # 判断是否因为停止符或其他终止条件而终止
        # done when reaches maximum length or LLM generates stop words
        if self.stop_str is not None and self.stop_str in self.action_history[-1]:
            terminated = True
        elif self._next_state_terminated[self.action_history[-1]]:
            terminated = True
        elif self.sep not in self.action_history[-1]:
            # This is because the output is stopped by eos
            # 输出被eos符号中断
            terminated = True
        else: terminated = False

        # 判断是否超出最大长度
        truncated = len(self.action_history) >= self.config["max_length"]
        assert len(self.action_history) <= self.config["max_length"] # 确保历史动作长度不超过最大长度
        if terminated or truncated:
            # 如果终止或超过最大长度，判断是否正确生成答案
            if self._is_correct(self.action_history[-1]):
                info["winner"] = 1  # 如果答案正确，则是赢家1
            else:
                info["winner"] = 2  # 否则是赢家2
            return terminated, truncated, info  # 返回是否终止、是否截断和信息

        return terminated, truncated, info  # 如果未终止或截断，返回相应状态

    def copy(self):
        # 该方法用于创建当前环境对象的副本（深拷贝）
        # 创建一个新环境对象，传递当前环境的配置信息以及其他必要的参数
        env = self.__class__(
            self.config,  # 当前环境的配置
            self.math_problems,  # 当前数学问题列表
            self.llm_gen_fn,  # 生成函数
            self._task_desc_str,  # 任务描述字符串
            self._cot_example_str,  # 示例字符串
            self._problem_format_str,  # 问题格式字符串
            reset=False,  # 不重置环境
        )
        
        # 使用深拷贝来复制环境中的重要属性
        env.math_problem = copy.deepcopy(self.math_problem)  # 复制当前的数学问题
        env._legal_actions = copy.deepcopy(self._legal_actions)  # 复制合法动作列表
        env.action_history = copy.deepcopy(self.action_history)  # 复制历史动作记录
        env._init_query = copy.deepcopy(self._init_query)  # 复制初始化查询字符串
        env._next_state_terminated = copy.deepcopy(self._next_state_terminated)  # 复制下一个状态是否终止的标志

        # 返回新的环境对象副本
        return env

    @property
    def legal_actions(self):
        # 返回当前环境对象的合法动作列表
        return self._legal_actions
