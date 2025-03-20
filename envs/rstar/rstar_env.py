"""
Wrapping the base environment with customised
update_legal_actions() function
"""

from heapq import merge
from typing import List, Dict, Tuple
import os
import numpy as np

from envs.MATH.env import CoTEnv
from envs.base_env import NoLegalActionException, ResetException
from tqdm import tqdm

from reason.inference.lm_call import LMCallingConfig
from .rstar_utils import *
from .eval_src.Evaluator import MATHEvaluator, QwenMATHEvaluator
from envs.MATH.prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP

from pathlib import Path

# Get the file path of the current script
# 获取当前脚本所在的目录路径
CURRENT_DIR = Path(__file__).parent


class IDCounter:
    def __init__(self):
        self.id = 0

    def count(self):
        self.id += 1  # 调用一次 ID加一
        return self.id


class Env(CoTEnv):
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
        """
        rStar call LLM inference in each of the nodes, checking current nodetype and do corresponding thinking
        OpenR apply LLM call in central Env entity (update_legal_action()), so we need to take some records...
        Args:
            config:
            math_problems:
            llm_gen_fn:
            task_desc_str:
            
            cot_example_str:
            problem_format_str:
            reset:
        """
        """
        初始化Env类，负责在每个节点调用LLM推理，检查当前节点类型并执行相应的思考。
        在OpenR环境中，LLM调用是在中心Env实体中执行的，因此需要进行一些记录。

        Args:
            config: 配置字典，包括生成配置等信息。
            math_problems: 数学问题的列表。
            llm_gen_fn: 用于生成LLM推理的函数。
            task_desc_str: 任务描述的字符串。
            cot_example_str: 示例字符串，用于Chain of Thought (CoT)。
            problem_format_str: 问题格式字符串。
            reset: 是否重置环境。
        """
        # 调用父类初始化方法，初始化相关配置
        super().__init__(
            config,
            math_problems,
            llm_gen_fn,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset,
        )

        # 初始化实例变量
        self.current_node_type = None        # 当前节点类型
        self.disable_a1 = False              # 是否禁用A1步骤
        self.disable_a5 = False              # 是否禁用A5步骤
        self.enable_potential_score = False  # 是否启用潜在分数

        # potential score is disable due to https://github.com/zhentingqi/rStar/issues/12
        # self.parent_is_subquestion = False          # TODO(yan): in rStar this seems to be false alltime, need to check

        # LLM generation config
        # LLM生成配置
        self.gen_cfg = config["generation_config"]

        # default parameter setting in original repo
        # 原始仓库中的默认参数设置
        self.num_a1_steps = 3  # A1步骤的数量
        self.mcts_num_last_votes = 32  # MCTS中最后投票的数量
        self.num_subquestions = 3  # 子问题的数量
        self.num_votes = 10  # 每个步骤的投票数量
        self.node_counter = IDCounter()  # 节点计数器（根节点） # root
        self.ost_new_tokens = 256  # 用于Ost生成的tokens数
        self.direct_answer_new_tokens = 1024  # 直接答案生成的tokens数
        self.subquestion_new_tokens1 = 128  # 子问题生成的tokens数1
        self.subquestion_new_tokens2 = 512  # 子问题生成的tokens数2
        self.rephrased_q_new_tokens = 512  # 重述问题生成的tokens数
        self.re_subanswer_new_tokens = 1024  # 重述答案生成的tokens数
        self.print_log = False  # 是否打印日志
        self.total_api_call_completion = 0  # 总的API调用次数
        self.total_tree_completion = 0  # 总的树形结构完成次数

        # self.task_name = "MATH"
        # 分隔符
        self.sep = "\n\n"
        self._init_query = None  # 初始化查询
        self._next_state_terminated = None  # 下一个状态是否已终止

        # loading template
        # 加载模板
        with open(
            os.path.join(
                CURRENT_DIR, f"prompts/MATH/decompose/decompose_template.json"
            ),
            "r",
        ) as f:
            decompose_template = json.load(f)
            self.question_index = decompose_template["index"] # 从模板中获取问题索引

        # 加载不同的提示（prompt）
        self.decompose_prompt = read_txt(
            os.path.join(CURRENT_DIR, f"prompts/MATH/decompose/decompose_prompt.txt")
        )  # 分解问题的prompt
        self.fewshot_cot_prompt = read_txt(
            os.path.join(
                CURRENT_DIR, f"prompts/MATH/fewshot_cot/fewshot_cot_prompt.txt"
            )
        )  # Few-shot CoT的prompt
        self.fewshot_cot_config = read_json(
            os.path.join(
                CURRENT_DIR, f"prompts/MATH/fewshot_cot/fewshot_cot_config.json"
            )
        )  # Few-shot CoT的配置

        # 如果没有禁用A1步骤
        # ost：Open-ended Step-by-step
        if not self.disable_a1:  # A1: Propose an one-step thought.
            self.fewshot_ost_prompt = read_txt(
                os.path.join(
                    CURRENT_DIR, f"prompts/MATH/fewshot_ost/fewshot_ost_prompt.txt"
                )
            )  # Few-shot Ost的prompt
            self.fewshot_ost_config = read_json(
                os.path.join(
                    CURRENT_DIR, f"prompts/MATH/fewshot_ost/fewshot_ost_config.json"
                )
            ) # Few-shot Ost的配置

        # 如果没有禁用A5步骤
        if not self.disable_a5:  # A5: Rephrase the question/sub-question.
            # 重述问题/子问题的prompt
            self.rephrasing_prompt_template = read_txt(
                os.path.join(
                    CURRENT_DIR, f"prompts/MATH/rephrasing_prompt_template.txt"
                )
            )
            # 加载重述问题后的分解prompt
            self.decompose_prompt_rephrased = read_txt(
                os.path.join(
                    CURRENT_DIR,
                    f"prompts/MATH/decompose/decompose_prompt_rephrased.txt",
                )
            )
            # 加载重述问题后的Few-shot CoT prompt
            self.fewshot_cot_prompt_rephrased = read_txt(
                os.path.join(
                    CURRENT_DIR,
                    f"prompts/MATH/fewshot_cot/fewshot_cot_prompt_rephrased.txt",
                )
            )
            # 加载重述问题后的Few-shot Ost prompt
            self.fewshot_ost_prompt_rephrased = read_txt(
                os.path.join(
                    CURRENT_DIR, f"prompts/MATH/fewshot_ost/fewshot_ost_prompt.txt"
                )
            )

        # load evaluator
        # 加载评估器
        self.evaluator = QwenMATHEvaluator()  # MATHEvaluator()

    def set_problem(self, idx):
        # 设置当前数学问题为列表中指定索引的问题
        self.math_problem = self.math_problems[idx]

    def reset(self, update_legal_action=True):
        # 重置当前问题，通常在某些操作失败后调用
        # 此方法默认会将当前问题设置为列表中的第一个问题
        self.set_problem(
            idx=0
        )  # retrive the first question set {'question': xxx, 'answer': xxx}
           # 获取第一个问题集 {'question': xxx, 'answer': xxx}

    @override
    def try_update_legal_action(self, node):
        # 尝试更新节点的合法动作，最多尝试3次
        cnt = 0
        while cnt < 3:
            cnt += 1
            try:
                # 尝试更新节点的合法动作
                updated_node = self.update_legal_actions(node)
                break
            except NoLegalActionException as e:
                # 如果遇到无合法动作异常，最多重试3次
                if cnt == 3:
                    # 如果3次尝试后仍然失败，则抛出重置异常
                    raise ResetException

        # info = {"api_completion_token": api_completion_token}
        return updated_node # 返回更新后的节点

    @override
    def update_legal_actions(self, current_node):
        """
        Think differently depending on current nodetype (status). The function directly create children node and
        add them to the parent node
        Returns:

        """
        """
        根据当前节点的类型执行不同的操作。该函数直接生成子节点并将它们添加到父节点。
        依据当前节点类型，提出相应的问题或采取行动。
        
        Returns:
            current_node.children: 当前节点的子节点列表
        """
        # 获取当前节点的类型
        current_node_type = current_node.node_type

        # depending on type of current node, ask corresponding question
        # 根据节点类型，决定采取的操作
        if current_node_type is Node_Type.USER_QUESTION:
            # 用户提问节点
            # A1: Propose an one-step thought.
            # A1: 提出一个单步思考
            if not current_node.disable_a1:
                self.do_action_generate_ost_step(current_node)

            # A2: Propose the remaining thought steps
            # A2: 提出剩余的思考步骤
            self.do_action_generate_direct_answers(current_node)

            # A3: Propose next sub-question along with its answer.
            # A3: 提出下一个子问题及其答案
            self.do_action_generate_subquestions(current_node)

            # A5: Rephrase the question/sub-question.
            # A5: 重新措辞问题或子问题
            if not current_node.disable_a5:
                self.do_action_generate_rephrased_user_question(current_node)

        elif current_node_type is Node_Type.REPHRASED_USER_QUESTION:
            # 重新措辞后的用户问题节点
            # A1: Propose an one-step thought.
            # A1: 提出一个单步思考
            if not current_node.disable_a1:
                self.do_action_generate_ost_step(current_node)

            # A2: Propose the remaining thought steps
            # A2: 提出剩余的思考步骤
            self.do_action_generate_direct_answers(current_node)

            # A3: Propose next sub-question along with its answer.
            # A3: 提出下一个子问题及其答案
            self.do_action_generate_subquestions(current_node)

        elif current_node_type is Node_Type.DIRECT_ANSWER:
            # 直接回答节点，不能生成子节点
            raise ValueError("DIRECT_ANSWER node cannot create children!!")

        elif current_node_type is Node_Type.SUBQUESTION:
            # 子问题节点
            # A1: Propose an one-step thought.
            # A1: 提出一个单步思考
            if not current_node.disable_a1:
                self.do_action_generate_ost_step(current_node)

            # A2: Propose the remaining thought steps
            # A2: 提出剩余的思考步骤
            self.do_action_generate_direct_answers(current_node)

            # A3: Propose next sub-question along with its answer.
            # A3: 提出下一个子问题及其答案
            self.do_action_generate_subquestions(current_node)

            # A4: Answer the sub-question again.
            # A4: 对子问题再次进行回答
            self.do_action_generate_re_subanswers(current_node)

        elif current_node_type is Node_Type.RE_SUBANSWER:
            # 重新回答的子问题节点
            # A1: Propose an one-step thought.
            # A1: 提出一个单步思考
            if not current_node.disable_a1:
                self.do_action_generate_ost_step(current_node)

            # A2: Propose the remaining thought steps
            # A2: 提出剩余的思考步骤
            self.do_action_generate_direct_answers(current_node)

            # A3: Propose next sub-question along with its answer.
            # A3: 提出下一个子问题及其答案
            self.do_action_generate_subquestions(current_node)

        elif current_node_type is Node_Type.OST_STEP:
            # 单步思考节点
            # A1: Propose an one-step thought.
            # A1: 提出一个单步思考
            if not current_node.disable_a1:
                self.do_action_generate_ost_step(current_node)

            # A2: Propose the remaining thought steps
            # A2: 提出剩余的思考步骤
            self.do_action_generate_direct_answers(current_node)

        # 返回当前节点的所有子节点
        return current_node.children  # a list of children

    def is_terminal(self, node):
        """
        判断当前节点是否为终止节点
        Args:
            node: 当前节点对象

        Returns:
            bool: 如果是终止节点，返回True，否则返回False
        """

        def is_valid_leaf_node(n):
            """
            判断给定节点是否是有效的叶节点
            只有 SUBQUESTION 类型或 DIRECT_ANSWER 类型的节点才能是有效的解答叶节点

            Args:
                n: 要检查的节点

            Returns:
                bool: 如果节点是有效的叶节点，则返回True，否则返回False
            """
            # 1. SUBQUESTION 类型节点，且其子问题已经到达终止状态
            # 2. DIRECT_ANSWER 类型节点
            # ! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
            return (
                n.node_type is Node_Type.SUBQUESTION
                and reach_terminal_subquestion(n.subquestion, n.user_question)
            ) or n.node_type is Node_Type.DIRECT_ANSWER

        # 判断当前节点是否已达到最大深度限制或者是否为有效叶节点
        return (node.depth >= node.max_depth_allowed) or is_valid_leaf_node(node)

    def do_action_generate_ost_step(self, node, parent_is_subquestion=False):
        """
        For current state, propose one-step thought, return legal action portion
        Args:
            parent_is_subquestion:

        Returns:
        """
        """
        针对当前状态，生成一步思考并返回合法的动作部分。
        这一步骤是解决问题时逐步推理的一个阶段。

        Args:
            node: 当前节点，表示思考的状态
            parent_is_subquestion: 是否为父节点是子问题的标志

        Returns:
            None
        """

        # 打印日志，提示正在生成一步思考步骤
        if self.print_log:
            print(f"---- Generating one-step thought steps for node ...")

        #! ACTION: generate one-step thought step
        # 初始化一个空的 OST 步骤列表
        ost_step_list = []
        # formating
        # 如果父节点是子问题，暂时没有实现相关逻辑，抛出异常
        if parent_is_subquestion:
            raise NotImplementedError  # 该分支似乎不会被执行 # this branche seems unreachable
            # 如果是子问题相关，可能会合并子问题的 OST 步骤
            # existing_ost_steps, next_ost_step_id = concat_subqs_subas_as_ost_steps(solution_trace)

        # 否则，获取现有的 OST 步骤和下一个步的 ID
        else:
            # 通过合并现有的 OST 步骤来生成新的步骤
            existing_ost_steps, next_ost_step_id = concat_ost_steps(node.solution_trace)

        # 根据模板生成输入提示
        io_input = (
            self.fewshot_ost_config["prompt_template"].format(
                examples=(
                    self.fewshot_ost_prompt
                    if not node.paraphrased
                    else self.fewshot_ost_prompt_rephrased
                ),
                instruction=node.user_question,
            )
            + existing_ost_steps
            + f"Step {next_ost_step_id}:"
        )
        # breakpoint()
        # 使用 LLM 生成模型调用，传递格式化后的输入
        io_output = self.llm_gen_fn(
            input_str=io_input,
            config=LMCallingConfig(
                n=self.num_a1_steps,  # 生成的步骤数量
                stop_str=["\n", "\n\n"],  # 设置停止标志 # check stopping token
                include_stop_str_in_output=True,
                max_new_tokens=self.ost_new_tokens, # 限制生成的最大新词数量
                **self.gen_cfg,  # 其他生成配置
            ),
        )
        # 获取生成的思考步骤，并去除首尾空白字符
        ost_step_list = [io_output.strip() for io_output in io_output.text]
        # 更新 API 调用的统计信息
        self.total_api_call_completion += io_output.completion_tokens
        self.total_tree_completion += sum(
            io_output.num_tokens
        )  # 累加树的 token 消耗 # incase of action post-processing

        # 为每一个生成的思考步骤，创建一个新的节点，并将其添加为当前节点的子节点
        potential_answers_list = [None] * len(ost_step_list) # 初始化潜在答案列表

        # 遍历生成的每个思考步骤，创建新的子节点
        for ost_step, potential_answers in zip(ost_step_list, potential_answers_list):
            node.children.append(
                RstarLanguageNode(
                    id=self.node_counter.count(), # 为节点分配唯一 ID
                    parent=node, # 当前节点作为父节点  # TODO[yan]: check over-nesting
                    depth=node.depth + 1, # 节点深度递增
                    node_type=Node_Type.OST_STEP, # 设置节点类型为 OST_STEP
                    ost_step=ost_step, # 当前思考步骤的内容
                )
            )
        # 无返回值，直接修改当前节点的子节点
        return

    def do_action_generate_direct_answers(self, node):
        """
        生成针对用户问题的直接回答，并将其作为子节点添加到当前节点。
        
        参数:
            node: 当前节点，包含了用户问题及相关信息
        """
        # 如果需要打印日志，输出正在生成直接回答的信息
        if self.print_log:
            print(f"---- Generating direct answers for node ...")
        # ! ACTION: generate direct answer for the user question (w/ or w/o hint)
         # 如果节点类型不是用户问题或重述用户问题，生成提示信息
        if (
            node.node_type is not Node_Type.USER_QUESTION
            and node.node_type is not Node_Type.REPHRASED_USER_QUESTION
        ):
            # 使用节点的解答轨迹生成提示
            hint = make_hint(node.solution_trace, node.node_type)
        else:
            hint = None  # 如果是用户问题或重述用户问题，则不生成提示

        # 初始化存储直接回答和信心值的列表
        direct_answer_list, value_list = [], []

        # 设置要生成的回答数量
        num_return = self.mcts_num_last_votes
        # 根据是否是重述用户问题选择不同的少量示例提示
        fewshot_cot_prompt = (
            self.fewshot_cot_prompt
            if not node.paraphrased
            else self.fewshot_cot_prompt_rephrased
        )
        # TODO: committed on Jan 18
        #question = node.user_question + "\n\n" + hint if hint is not None else ""
        # 构造要询问的完整问题，包括用户问题和可能的提示
        question = node.user_question + "\n\n" + (hint if hint is not None else "")
        # 使用格式模板生成输入提示
        io_input = self.fewshot_cot_config["prompt_template"].format(
            examples=fewshot_cot_prompt, instruction=question
        )
        # breakpoint()
        # 调用 LLM 生成直接回答
        io_output = self.llm_gen_fn(
            input_str=io_input,
            config=LMCallingConfig(
                n=num_return,  # 设置生成的回答数量
                max_new_tokens=self.direct_answer_new_tokens,  # 设置生成的最大 token 数
                include_stop_str_in_output=True,  # 包括停止标记
                stop_str=self.fewshot_cot_config["stop_tokens"],  # 停止标记
                **self.gen_cfg,  # 其他生成配置
            ),
        )

        # 更新 API 调用的统计信息
        self.total_api_call_completion += io_output.completion_tokens
        self.total_tree_completion += sum(
            io_output.num_tokens
        )  # 更新树的 token 消耗  # incase of action post-processing

        # 清理生成的输出，去除首尾的空白字符
        cleaned_io_output_list = [
            io_output.strip() for io_output in io_output.text
        ]  # ! cleaning

        try:
            # 确保生成的输出列表不为空
            assert len(cleaned_io_output_list) > 0

            # 如果只有一个输出，直接使用该输出作为最可能的答案
            if len(cleaned_io_output_list) == 1:
                most_likely_answer = cleaned_io_output_list[0]
                likelihood = 1
            else:
                # 如果有多个输出，使用评估器找到最有信心的回答
                _, most_likely_answer, _, likelihood = (
                    self.evaluator.find_most_confident_answer(cleaned_io_output_list)
                )
                assert likelihood > 0  # 确保信心值大于 0

        except Exception as e:
            # 如果处理过程中出现异常，抛出生成器错误并提供详细信息
            raise GeneratorError(
                source="generate direct answer from: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )

        # 将最有信心的回答和信心值添加到列表
        direct_answer_list.append(most_likely_answer)
        value_list.append(likelihood)

        # 为每一个直接回答和对应的信心值创建子节点并添加到当前节点的子节点列表中
        for direct_answer, value in zip(direct_answer_list, value_list):
             # 如果信心值是无效的（如 NaN 或小于等于 0），则抛出未实现的错误
            if np.isnan(value) or value <= 0:
                raise NotImplementedError
            
            # 创建新的子节点并添加到当前节点的子节点列表
            node.children.append(
                RstarLanguageNode(
                    id=self.node_counter.count(),  # 为新节点分配唯一 ID
                    parent=node,  # 当前节点作为父节点
                    depth=node.depth + 1,  # 节点深度递增
                    node_type=Node_Type.DIRECT_ANSWER,  # 设置节点类型为 DIRECT_ANSWER
                    node_value=value,  # 设置节点的值为回答的信心值
                    direct_answer=direct_answer,  # 设置节点的直接回答
                )
            )

        return

    def do_action_generate_subquestions(self, node):
        # 如果打印日志为真，则输出日志
        if self.print_log:
            print(f"---- Generating subquestions for node ...")

        # 初始化子问题、子答案和值的列表
        subquestion_list, subanswer_list, value_list = [], [], []
        # 选择分解提示词，取决于是否对节点的问题进行了重述
        decompose_prompt = (
            self.decompose_prompt
            if not node.paraphrased
            else self.decompose_prompt_rephrased
        )

        # ! generate subquestions
        # ! 生成子问题
        # 获取现有的子问题和子答案，并确定下一个子问题的ID
        existing_subquestions_and_subanswers, next_subquestion_id = (
            concat_subqs_and_subas(node.solution_trace, self.question_index)
        )
        # 构造输入字符串（包括问题和现有的子问题与子答案）
        io_input = (
            decompose_prompt
            + "\n\n"
            + f"Question {self.question_index}: {node.user_question}"
            + "\n"
            + existing_subquestions_and_subanswers
            + f"Question {self.question_index}.{next_subquestion_id}:"
        )
        # breakpoint()
        # 调用模型生成子问题
        io_output = self.llm_gen_fn(
            input_str=io_input,
            config=LMCallingConfig(
                n=self.num_subquestions,
                max_new_tokens=self.subquestion_new_tokens1,
                include_stop_str_in_output=True,
                stop_str=[
                    "\n",
                    "\n\n",
                    "Answer",
                    "Answer ",
                    f"Answer {self.question_index}.{next_subquestion_id}",
                    f"Answer {self.question_index}.{next_subquestion_id}:",
                    f"Answer {self.question_index}.{next_subquestion_id}: ",
                ],
                **self.gen_cfg,
            ),
        )
        # 累加API调用的token数量
        self.total_api_call_completion += io_output.completion_tokens
        self.total_tree_completion += sum(
            io_output.num_tokens
        ) # 更新树的token数量（用于后期处理） # incase of action post-processing

        # 提取并清理生成的子问题
        subquestion_list = [o.strip() for o in io_output.text]

        # ! generate subanswers to the subquestions generated above
        # ! 生成子问题的答案
        io_input_list = []
        for subquestion in subquestion_list:
            # 构建每个子问题的输入字符串
            io_input = (
                decompose_prompt
                + "\n\n"
                + f"Question {self.question_index}: {node.user_question}"
                + "\n"
                + existing_subquestions_and_subanswers
                + f"Question {self.question_index}.{next_subquestion_id}: "
                + subquestion
                + "\n"
                + f"Answer {self.question_index}.{next_subquestion_id}:"
            )
            io_input_list.append(io_input)

        # 判断是否达到终止子问题的条件
        if reach_terminal_subquestion(
            subquestion=subquestion, user_question=node.user_question
        ):
            num_return = self.mcts_num_last_votes  # 返回最后投票的数量
        else:
            num_return = self.num_votes  # 默认返回的投票数量

        io_output_list = []
        # 为每个子问题生成答案
        for i in io_input_list:
            # breakpoint()
            # 调用模型生成子问题的答案
            _each_output = self.llm_gen_fn(
                input_str=i,
                config=LMCallingConfig(
                    n=num_return,
                    max_new_tokens=self.subquestion_new_tokens2,
                    include_stop_str_in_output=True,
                    stop_str=[
                        "\n",
                        "\n\n",
                        f"Question {self.question_index}.{next_subquestion_id + 1}",
                    ],
                    **self.gen_cfg,
                ),
            )
            # 存储每个子问题的输出
            io_output_list.append(_each_output.text)
            # 累加API调用的token数量
            self.total_api_call_completion += _each_output.completion_tokens
            self.total_tree_completion += sum(
                _each_output.num_tokens
            )  # 更新树的token数量（用于后期处理）  # incase of action post-processing

        # 清理每个输出的文本
        cleaned_io_output_list = [
            [io_output.strip() for io_output in io_output_group]
            for io_output_group in io_output_list
        ]

        # completion_tokens = io_output_list.completion_tokens

        # 对每个子问题的输出进行后处理
        for i, cleaned_io_output_group in enumerate(cleaned_io_output_list):
            try:
                # 获取最有可能的答案及其概率
                most_likely_answer, likelihood = self._get_most_likely_answer(
                    cleaned_io_output_group
                )
            except Exception as e:
                # 如果处理错误，抛出自定义异常
                raise GeneratorError(
                    source="generate answer to subquestions",
                    io_input=io_input_list[i],
                    io_output_list=cleaned_io_output_group,
                )
            # 将最有可能的答案和它的值添加到答案列表和权重列表
            subanswer_list.append(most_likely_answer)
            value_list.append(likelihood)
        # 确保子问题、答案和值的数量一致
        assert len(subquestion_list) == len(subanswer_list) == len(value_list)
        # 初始化潜在答案列表
        potential_answers_list = [None] * len(subquestion_list)

        # 将每个子问题、子答案、权重以及潜在答案进行封装
        for subquestion, subanswer, value, potential_answers in zip(
            subquestion_list, subanswer_list, value_list, potential_answers_list
        ):
            # 如果值为NaN或小于等于0，设为一个很小的值
            if np.isnan(value) or value <= 0:
                value = 0.01
                # breakpoint()
            # 将子问题和答案作为子节点添加到树中
            node.children.append(
                RstarLanguageNode(
                    id=self.node_counter.count(),
                    parent=node,
                    depth=node.depth + 1,
                    node_type=Node_Type.SUBQUESTION,
                    node_value=value,
                    subquestion=subquestion,
                    subanswer=subanswer,
                    is_new_subquestion=True,
                )
            )

        return

    def do_action_generate_rephrased_user_question(self, node):
        if self.print_log:
            print(f"---- Generating rephrased user question for node ...")

        rephrased_user_question_list = [] # 存储重新表述的用户问题
        io_input = self.rephrasing_prompt_template  # 获取重述的提示模板
        io_input += "\n\n"
        io_input += "Original Question: " + node.user_question + "\n" # 加入原始问题
        io_input += "Rephrased Question: Given a list of conditions, please answer the question. Condition 1: "
        # breakpoint()
        _io_output = self.llm_gen_fn(
            input_str=io_input,
            config=LMCallingConfig(
                n=1,
                max_new_tokens=self.rephrased_q_new_tokens, # 最大生成的令牌数
                include_stop_str_in_output=True,
                stop_str=["\n", "\n\n"],  # 停止字符串的设置
                **self.gen_cfg,  # 其它生成配置
            ),
        )
        io_output = _io_output.text
        self.total_api_call_completion += _io_output.completion_tokens # 更新API调用的完成令牌数
        self.total_tree_completion += sum(
            _io_output.num_tokens
        )  # 更新树形结构的完成令牌数（处理后续操作）  # incase of action post-processing

        assert len(io_output) == 1 # 确保输出长度为1
        io_output = (
            "Given a list of conditions, please answer the question. Condition 1: "
            + io_output[0]
        )  # 拼接生成的重新表述问题
        rephrased_user_question_list.append(io_output)  # 将重新表述的问题添加到列表

        potential_answers_list = [None] * len(rephrased_user_question_list)  # 初始化潜在答案列表

        # creating children
        # 创建子节点
        for rephrased_user_question, potential_answers in zip(
            rephrased_user_question_list, potential_answers_list
        ):
            node.children.append(
                RstarLanguageNode(
                    id=self.node_counter.count(),
                    parent=node, # 设置父节点
                    depth=node.depth + 1, # 设置子节点的深度
                    node_type=Node_Type.REPHRASED_USER_QUESTION,  # 节点类型
                    rephrased_user_question=rephrased_user_question, # 设置重新表述的问题
                )
            )

        return

    def do_action_generate_re_subanswers(self, node):
        if self.print_log:
            print(f"---- Generating re-subanswers for node ...")
        re_subanswer_list, value_list = [], []  # 存储重新回答和对应的值

        user_question_context, _ = split_user_question(node.user_question)  # 分割用户问题

        last_subquestion_id = int(sorted(node.solution_trace.keys())[-1]) # 获取最后一个子问题的ID
        last_subquestion = node.solution_trace[last_subquestion_id]["subquestion"] # 获取最后一个子问题
        # ! few shot cot
        # 进行少样本链式推理（Few-Shot COT）
        question = (
            f"{user_question_context} {last_subquestion}"
            if not node.paraphrased
            else f"{user_question_context} Question: {last_subquestion}"
        )
        fewshot_cot_prompt = (
            self.fewshot_cot_prompt
            if not node.paraphrased
            else self.fewshot_cot_prompt_rephrased
        )
        question += "\n\n"   # 在问题后加空行（提示符） # hint is None
        io_input = self.fewshot_cot_config["prompt_template"].format(
            examples=fewshot_cot_prompt, instruction=question  # 填充少样本推理模板
        )
        # breakpoint()
        io_output = self.llm_gen_fn(
            input_str=io_input,
            config=LMCallingConfig(
                n=self.num_votes, # 设置投票数
                max_new_tokens=self.re_subanswer_new_tokens,  # 最大生成的令牌数
                include_stop_str_in_output=True,
                stop_str=self.fewshot_cot_config["stop_tokens"],  # 停止标记
                **self.gen_cfg,  # 其它生成配置
            ),
        )
        self.total_api_call_completion += io_output.completion_tokens  # 更新API调用的完成令牌数
        self.total_tree_completion += sum(
            io_output.num_tokens
        )  # 更新树形结构的完成令牌数（处理后续操作） # incase of action post-processing

        cleaned_io_output_list = [
            io_output.strip() for io_output in io_output.text
        ]  # 清理输出文本（去除空白符）  # ! cleaning
        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(
                cleaned_io_output_list  # 获取最可能的答案及其可能性
            )
        except Exception as e:
            raise GeneratorError(
                source="generate re-subanswers: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )
        re_subanswer_list.append(most_likely_answer)  # 将最可能的答案添加到列表
        value_list.append(likelihood)  # 将答案的可能性值添加到列表
        potential_answers_list = [None] * len(re_subanswer_list)  # 初始化潜在答案列表

        # creating children
        # 创建子节点
        for re_subanswer, value, potential_answers in zip(
            re_subanswer_list, value_list, potential_answers_list
        ):
            if np.isnan(value) or value <= 0: # 检查可能性值是否合法
                breakpoint() # 如果值非法，可以通过断点调试
            node.children.append(
                RstarLanguageNode(
                    id=self.node_counter.count(),
                    parent=node,   # 设置父节点 # check node chck node, do we need to pass children as well?
                    depth=node.depth + 1, # 设置子节点的深度
                    node_type=Node_Type.RE_SUBANSWER,  # 节点类型
                    node_value=value,  # 设置子问题的可能性值
                    re_subanswer=re_subanswer,  # 设置重新回答的内容
                )
            )

        return

    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        # 确保输入的输出列表不为空
        assert len(io_output_list) > 0

        # 如果只有一个输出，直接将其视为最有可能的答案，并且置信度为1
        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0] # 最有可能的答案是唯一的输出
            confidence = 1 # 置信度为1（因为只有一个输出，肯定是最有可能的）
        else:
            # 如果有多个输出，则通过评估器来找出最有可能的答案
            _, most_confident_answer_full_completion, _, confidence = (
                self.evaluator.find_most_confident_answer(io_output_list) # 调用评估器来找到最有信心的答案
            )
            # 确保置信度大于0，表示找到了有信心的答案
            assert confidence > 0

        # 返回最有可能的答案和对应的置信度
        return most_confident_answer_full_completion, confidence
