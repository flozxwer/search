from dataclasses import dataclass
import functools
from typing import Dict
from reason.inference.lm_call import LMCallingConfig, LanguageModelCallingFunction
from reason.inference.rm_call import RewardModelCallingFunction
from reason.evaluation.evaluator import SolutionOutput, Task, TreeSearchSolutionOutput
from reason.guided_search.tree import SearchTree
from reason.guided_search.rstar import RstarSearchTree


@dataclass
class BasicConfig:
    task_name: str  # 配置中的任务名称


@dataclass
class CoTConfig(BasicConfig):
    pass  # CoTConfig 继承 BasicConfig，目前没有额外属性


def cot(
    config: CoTConfig,
    gen_config: LMCallingConfig,
    problem_inst: Dict[str, str],
    llm_call: LanguageModelCallingFunction,
    rm_call: RewardModelCallingFunction,
) -> SolutionOutput:
    # 设置生成配置（gen_config）的一些默认参数
    gen_config = LMCallingConfig(
        n=1,            # 生成的数量
        temperature=0,  # 温度参数（决定采样的多样性，0表示最确定的答案）
        top_k=1,        # 限制从前top_k个词中选择
        top_p=1.0,      # 限制采样的累计概率
        max_new_tokens=gen_config.max_new_tokens,  # 使用传入的max_new_tokens
    )
    # 设置 config 的 num_sequence 为 1（生成序列数）
    config.num_sequence = 1
    # 调用 `best_of_n` 函数来生成最优的解答，并返回结果
    return best_of_n(config, gen_config, problem_inst, llm_call, rm_call)


# 配置类，继承自 BasicConfig，增加了一个 num_sequence 参数
@dataclass
class BestOfNConfig(BasicConfig):
    num_sequence: int = 32


def best_of_n(
    config: BestOfNConfig,                  # BestOfNConfig 配置对象
    gen_config: LMCallingConfig,            # 生成配置对象
    problem_inst: Dict[str, str],           # 问题实例，包含问题等信息
    lm_call: LanguageModelCallingFunction,  # 调用语言模型的函数
    rm_call: RewardModelCallingFunction,    # 调用奖励模型的函数
) -> SolutionOutput:
    # 如果生成配置的 max_new_tokens 小于 256，则发出警告
    if gen_config.max_new_tokens < 256:
        print("Warning: max_new_tokens is less than 256")

    gen_config.n = config.num_sequence
    task = Task(task_name=config.task_name) # 创建任务对象
    prompt = task.prompt_fn(problem_inst["question"]) # 使用任务的提示函数生成任务的提示
    output = lm_call(prompt, gen_config)
    completion_tokens = output.num_tokens  # 获取生成结果的token数
    return SolutionOutput(
        solutions=output.text,   # 生成的文本
        completion_tokens=completion_tokens,
    )


@dataclass
class TreeSearchConfig(BasicConfig):
    # construction config
    tree_max_width: int = 10  # 树的最大宽度，决定每层最多有多少个子节点
    tree_max_depth: int = 10  # 树的最大深度，决定树的最大层数
    # node config
    init_critic_value: bool = True  # 是否初始化节点的评估值（默认初始化）

    def __post_init__(self):
        # 确保树的最大宽度和深度大于0
        assert self.tree_max_width > 0, \
            "Tree width must be greater than 0"
        assert self.tree_max_depth > 0, \
            "Tree depth must be greater than 0"

@dataclass
class BeamSearchConfig(TreeSearchConfig):
    beam_size: int = 1  # Beam search的大小，即每次扩展时保留的最佳候选数量

    def __post_init__(self):
        # 调用父类的__post_init__方法，验证树搜索的配置
        super().__post_init__()
        # 验证beam_size必须大于0
        assert self.beam_size > 0, \
            "Beam size must be greater than 0"
        # 验证初始化评估值的配置
        assert self.init_critic_value, \
            "BeamSearch should set init_critic_value to True"

def beam_search(
    config: BeamSearchConfig,               # 搜索配置，包括beam_size等
    gen_config: LMCallingConfig,            # 语言模型生成的配置
    problem_inst: Dict[str, str],           # 包含问题和答案的字典
    lm_call: LanguageModelCallingFunction,  # 语言模型调用函数
    rm_call: RewardModelCallingFunction,    # 奖励模型调用函数
) -> SolutionOutput:
    # 创建任务实例
    task = Task(task_name=config.task_name)
    # 初始化环境，传入生成配置和问题数据
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,             # 设置最大动作数（树的宽度）
            "max_length": config.tree_max_depth,              # 设置最大长度（树的深度）
            "stop_str": "The answer is ",                     # 设置停止字符串
            "generation_config": {
                "max_new_tokens": gen_config.max_new_tokens,  # 最大生成token数
                "temperature": gen_config.temperature,        # 生成的温度值（控制生成的随机性）
                "top_p": gen_config.top_p,                    # nucleus sampling的概率
                "top_k": gen_config.top_k,                    # top-k采样数
            },
        },
        # 提供数学问题和答案
        math_problems=[
            {
                "question": problem_inst["question"],  # 问题
                "answer": task.extract_groundtruth(problem_inst["answer"]),  # 真实答案
            }
        ],
        llm_gen_fn=lm_call,  # 语言模型生成函数
        # TODO(ziyu): set sep by lm_call.lm_step_tag
    )
    # 初始化搜索树实例
    search_tree = SearchTree(cfg={})

    # 创建奖励模型函数的部分应用，传递语言模型步长标记
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    
    # 执行beam search并获取轨迹列表
    traj_list = search_tree.beam_search(
        env, config.beam_size, config.tree_max_depth, rm_call_fn
    )

    # 返回搜索结果，包括文本、API完成tokens和树完成tokens
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],  # 解答文本
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],  # 每个解答的API完成tokens
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],  # 每个解答的树完成tokens
    )

@dataclass
class MCTSBaseConfig(TreeSearchConfig):
    # PUCT hparams
    # PUCT算法的超参数：用于平衡探索与利用的因子
    pb_c_base: float = 19652  # PUCT中的基础常数
    pb_c_init: float = 1.25   # PUCT中的初始化常数

@dataclass
class VanilaMCTSConfig(MCTSBaseConfig):
    # rollout step strategy, if `select_by_prior` is False,
    #  then select by the initial critic value
    # otherwise, random choice by the prior probability
    # 在rollout步骤中选择策略：
    # 如果 `select_by_prior` 为 False，则根据初始评估值选择
    # 否则，按先验概率随机选择
    select_by_prior: bool = False  
    # 每次路径搜索的数量
    num_path: int = 1
    
    def __post_init__(self):
        # 调用父类的初始化方法
        super().__post_init__()
        # 如果 `select_by_prior` 为 False，必须设置 `init_critic_value` 为 True
        if not self.select_by_prior:
            assert self.init_critic_value, \
                "VanilaMCTS with greedy as rollout method should set init_critic_value to True"
        # `num_path` 必须大于0
        assert self.num_path > 0

def vanila_mcts(
    config: VanilaMCTSConfig,               # MCTS的配置，包括rollout步骤和路径数等
    gen_config: LMCallingConfig,            # 语言模型生成配置
    problem_inst: Dict[str, str],           # 包含问题和答案的字典
    lm_call: LanguageModelCallingFunction,  # 语言模型调用函数
    rm_call: RewardModelCallingFunction     # 奖励模型调用函数
):
    # 创建任务实例
    task = Task(task_name=config.task_name)
    # 初始化环境，传入生成配置和问题数据
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,               # 设置最大动作数（树的宽度）
            "max_length": config.tree_max_depth,                # 设置最大深度（树的深度）
            "stop_str": "The answer is ",                       # 设置停止字符串
            "generation_config": {
                "max_new_tokens": gen_config.max_new_tokens,    # 最大生成token数
                "temperature": gen_config.temperature,          # 控制生成随机性的温度值
                "top_p": gen_config.top_p,                      # nucleus sampling的概率
                "top_k": gen_config.top_k,                      # top-k采样数
            },
        },
        math_problems=[
            {
                "question": problem_inst["question"],  # 问题
                "answer": task.extract_groundtruth(problem_inst["answer"]),  # 真实答案
            }
        ],
        llm_gen_fn=lm_call,  # 语言模型生成函数
    )

    # 初始化搜索树实例，传入PUCT的超参数和初始化评估值
    search_tree = SearchTree(
        cfg={
            "pb_c_base": config.pb_c_base,
            "pb_c_init": config.pb_c_init,
            "init_critic_value": config.init_critic_value,
        }
    )
    # 使用部分应用函数将奖励模型调用函数与语言模型步长标记绑定
    # rm_call_fn 是一个新创建的函数，它与 rm_call 函数具有相同的行为
    # 只是 lm_step_tag 参数已经被固定好了
    # 调用 rm_call_fn 时，lm_step_tag 会自动传入，而不需要在调用时显式提供
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    # 执行Vanila MCTS算法并获取解答路径列表
    traj_list = search_tree.vanila_mcts(
        simulate_env=env,                       # 环境实例
        num_path=config.num_path,               # 搜索的路径数
        reward_model_fn=rm_call_fn,             # 奖励模型函数
        select_by_prior=config.select_by_prior  # 是否根据先验概率选择
    )

    # 返回搜索结果，包括文本、API完成tokens和树完成tokens
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],  # 解答文本
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],   # 每个解答的API完成tokens
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],   # 每个解答的树完成tokens
    )


@dataclass
class RStarMCTSConfig(MCTSBaseConfig):
    # rollout step strategy, if `select_by_prior` is False,
    #  then select by the initial critic value
    # otherwise, random choice by the prior probability
    # rollout步骤策略：同VanilaMCTS
    select_by_prior: bool = False  # 是否根据先验概率选择
    num_path: int = 1  # 每次路径搜索的数量

    def __post_init__(self):
        super().__post_init__()
        # 如果 `select_by_prior` 为 False，必须设置 `init_critic_value` 为 True
        if not self.select_by_prior:
            assert self.init_critic_value, \
                "VanilaMCTS with greedy as rollout method should set init_critic_value to True"
        # `num_path` 必须大于0
        assert self.num_path > 0

def rstar_mcts(
        config: RStarMCTSConfig,                # RStarMCTS的配置，包含PUCT超参数等
        gen_config: LMCallingConfig,            # 语言模型生成配置
        problem_inst: Dict[str, str],           # 包含问题和答案的字典
        lm_call: LanguageModelCallingFunction,  # 语言模型调用函数
        rm_call: RewardModelCallingFunction     # 奖励模型调用函数
):
    # 创建任务实例，传入任务名称
    task = Task(task_name=config.task_name)
    # 初始化环境，传入生成配置和问题数据
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,       # 设置最大动作数（树的宽度）
            "max_length": config.tree_max_depth,        # 设置最大深度（树的深度）
            "stop_str": "The answer is ",               # 设置停止字符串
            "generation_config": {
                "temperature": gen_config.temperature,  # 控制生成随机性的温度值
                "top_p": gen_config.top_p,              # nucleus sampling的概率
                "top_k": gen_config.top_k,              # this is fixed for each llm call # top-k采样数
            },
        },
        math_problems=[
            {
                "question": problem_inst["question"],  # 问题
                "answer": task.extract_groundtruth(problem_inst["answer"]),  # 真实答案
            }
        ],
        llm_gen_fn=lm_call,  # 语言模型生成函数
    )

    # 初始化RStar搜索树实例，传入PUCT超参数和初始评估值
    search_tree = RstarSearchTree(
        cfg={
            "pb_c_base": config.pb_c_base,
            "pb_c_init": config.pb_c_init,
            "init_critic_value": config.init_critic_value,
        }
    )
    # 创建奖励模型函数，绑定语言模型步长标记
    rm_call_fn = functools.partial(rm_call, lm_step_tag=lm_call.lm_step_tag)
    # 执行RStar MCTS算法，获取解答路径列表
    traj_list = search_tree.rstar_mcts(
         simulate_env=env,                      # 环境实例
        num_path=config.num_path,               # 搜索的路径数
        reward_model_fn=rm_call_fn,             # 奖励模型函数
        select_by_prior=config.select_by_prior  # 是否根据先验概率选择
    )
    # 返回搜索结果，包括文本、API完成tokens和树完成tokens
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],  # 解答文本
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],  # 每个解答的API完成tokens
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],  # 每个解答的树完成tokens
    )
