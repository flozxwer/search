"""
The Node and MCTS class for AlphaZero.
"""

#
import copy
import json
import math

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
from distributed.utils import print_rank_0, print_with_rank
from envs.base_env import CoTEnv
import pdb
from tqdm import tqdm
import heapq
from loguru import logger


class Node(object):
    """
    Overview:
        The node base class for tree_search.
        树搜索的节点基类
    """

    def __init__(
        self, parent: "Node" = None, prior_p: float = 1.0, initial_value: float = 0.0
    ) -> None:
        self._parent = parent        # 父节点
        self._children = {}          # 子节点字典
        self._visit_count = 0        # 访问次数
        self._value_sum = 0          # 节点的值之和
        self.prior_p = prior_p       # 节点的先验概率
        self.prior_p_ori = prior_p   # 初始先验概率

        self._initial_value = initial_value  # 初始值
        self._terminated = False     # 是否终止节点

    def __lt__(self, other):
        # 比较节点的初始值，用于排序
        return self._initial_value < other._initial_value

    @property
    def terminated(self):
        # 返回节点是否为终止节点
        return self._terminated

    def set_as_terminate_node(self):
        # 设置当前节点为终止节点
        self._terminated = True

    @property
    def value(self) -> float:
        """
        Overview:
            The value of the current node.
            返回当前节点的值，用于计算 UCB 分数
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
            当前节点的值：如果未访问，则返回初始值；否则，返回值的平均数
        """
        if self._visit_count == 0:
            # if not visited, return the initial value
            # 如果未访问，返回初始值
            return self._initial_value
        return self._value_sum / self._visit_count

    def update(self, value: float) -> None:
        """
        Overview:
            Updata the current node information, such as visit_count and value_sum.
            更新当前节点的访问次数和节点值之和
        Arguments:
            - value (:obj:`Int`): The value of the node.
            - value: 当前节点的值
        """
        self._visit_count += 1
        self._value_sum += value

    def update_recursive(self, leaf_value: float, mcts_mode: str) -> None:
        """
        Overview:
            Update node information recursively.
            递归地更新节点信息
        Arguments:
            - leaf_value (:obj:`Int`): The value of the node.
            - leaf_value: 当前节点的值
            - mcts_mode: 表示不同模式（如自对弈模式或与机器人对弈模式）
        """
        if mcts_mode == "self_play_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(-leaf_value, mcts_mode)
        if mcts_mode == "play_with_bot_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(leaf_value, mcts_mode)

    def is_leaf(self) -> Dict:
        """
        Overview:
            Check if the current node is a leaf node or not.
            检查当前节点是否为叶子节点
        Returns:
            - output (:obj:`Dict`): Dict type children node.
            - 返回子节点字典，若为空则是叶子节点
        """
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
            检查当前节点是否为根节点
        Returns:
            - output (:obj:`Bool`): Whether it is the parent node.
            - 返回布尔值，若为根节点则为True
        """
        return self._parent is None

    @property
    def parent(self) -> None:
        # 返回当前节点的父节点
        return self._parent

    @property
    def children(self) -> None:
        # 返回当前节点的子节点
        return self._children

    @property
    def visit_count(self) -> None:
        # 返回当前节点的访问次数
        return self._visit_count

    def get_info(self):
        # return [
        #     "visit_cnt: {}, value: {:.6f}, prior: {:.6f}".format(
        #         self.visit_count, self.value, self.prior_p)
        # ]
        # 返回当前节点的基本信息
        return {
            "visit_cnt": self.visit_count,
            "value": self.value,
            "prior_p": float(self.prior_p_ori),
            "initial_value": self._initial_value,
            "terminated": self.terminated,
        }

    def clear(self):
        # 重置节点的访问次数、值和优先概率
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = self.prior_p_ori

    def to_json(self):
        # 将当前节点及其子节点转化为 JSON 格式
        childrens = {}
        for name, child_node in self.children.items():
            childrens[name] = child_node.to_json()

        rets = {"children": childrens, "info": self.get_info()}
        return rets
    
    def __str__(self) -> str:
        # 输出当前节点的字符串表示
        if self.is_root():
            return "root" # 根节点
        else:
            return "child: value: {:.3f}, prior: {:.3f}".format(
                self.last_action, self.value, self.prior_p
            )


class LanguageNode(Node):
    # 语言节点类，继承自 Node 类

    text_state: Optional[str] = None  # 当前文本状态
    last_action: Optional[str] = None  # 上一步动作
    num_generated_token: Optional[int] = None  # 生成的 token 数量

    def __init__(
        self,
        parent: Node = None,                        # 父节点
        prior_p: float = 1.0,                       # 节点的优先概率
        prm_value: Optional[float] = None,          # 参数值
        text_state: Optional[str] = None,           # 文本状态
        last_action: Optional[str] = None,          # 上一步动作
        initial_value: float = 0.0,                 # 初始值
        num_generated_token: Optional[int] = None,  # 生成的 token 数量
    ) -> None:
        # 初始化节点，调用父类构造函数并设置额外属性
        super().__init__(parent, prior_p, initial_value)
        self.text_state = text_state
        self.last_action = last_action
        self.prm_value = prm_value
        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False  # 是否收集到 token 数量

    def get_path(self):
        # 获取从当前节点到根节点的路径（动作序列）
        ans = []
        node = self
        while not node.is_root():
            ans.append(node.last_action)  # 收集每一步的动作
            node = node.parent  # 移动到父节点
        return "\n".join(reversed(ans))  # 返回反向的路径

    def get_info(self):
        # 获取当前节点的详细信息
        info_dict = super().get_info()  # 获取父类的基本信息
        if not self.is_root():
            info_dict["last_action"] = self.last_action  # 添加上一步动作
            info_dict["prm_value"] = self.prm_value  # 添加参数值
        else:
            info_dict["text_state"] = self.text_state  # 如果是根节点，添加文本状态
        return info_dict
    
    def __str__(self):
        # 返回当前节点的字符串表示
        if self.is_root():
            return "root: {}".format(self.text_state)  # 根节点时显示文本状态
        else: 
            return "action: {}, value: {:.3f}, prior: {:.3f}".format(
                self.last_action, self.value, self.prior_p
            )  # 显示动作、值和优先概率



def get_root(node: Node):
    # 递归地寻找并返回给定节点的根节点
    while not node.is_root():  # 当节点不是根节点时
        node = node.parent  # 移动到父节点
    return node  # 返回根节点


class SearchTree:
    """
    Overview:
        MCTS search process.
        MCTS（蒙特卡罗树搜索）过程
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg

        self._num_simulations = self._cfg.get("num_simulations", 20) # 模拟次数

        # UCB formula
        # UCB公式的参数
        self._pb_c_base = self._cfg.get("pb_c_base", 19652)  # 19652 # UCB的基本常数
        self._pb_c_init = self._cfg.get("pb_c_init", 1.25)  # 1.25 # UCB的初始化常数

        # Root prior exploration noise.
        # 根节点的Dirichlet噪声
        self._root_dirichlet_alpha = self._cfg.get(
            "root_dirichlet_alpha", 0.3
        )  # 0.3  # for chess, 0.03 for Go and 0.15 for shogi. # 根节点的Dirichlet噪声（棋类不同游戏有不同的值）
        self._root_noise_weight = self._cfg.get("root_noise_weight", 0.25)  # 0.25 # 根节点噪声权重

        self.root = None  # 根节点

        self.answers = set()  # 正确答案集合
        self.wrong_answers = set()  # 错误答案集合
        self.visited_paths = None  # 已访问路径

        # 不使用终止奖励标志
        self.no_terminal_reward = self._cfg.get("no_terminal_reward", True)
        # 是否遮蔽非终止节点的值
        self.mask_non_terminal_node_value = self._cfg.get(
            "mask_non_terminal_node_value", False
        )

        # 是否初始化评论值
        self._init_critic_value = self._cfg.get("init_critic_value", True)

        self._completion_tokens = 0 # 完成的tokens数

    @property
    def num_generated_token(self):
        # 返回生成的token数
        return self._completion_tokens

    def clear_node(self, node):
        # 清除节点及其所有子节点的数据
        assert node is not None # 确保节点不为空
        node.clear() # 清除当前节点数据
        for child in node.children.values():
            self.clear_node(child) # 递归清除所有子节点

    def get_next_action(
        self,
        simulate_env: Type[CoTEnv],
        reward_fn: Optional[Callable] = None,
        temperature: int = 1.0,
        sample: bool = True,
        return_tree=False,
    ) -> Tuple[int, List[float]]:
        """
        Overview:
            calculate the move probabilities based on visit counts at the root node.
            根据根节点的访问次数计算每个动作的概率，并根据这些概率选择下一个动作。
        Arguments:
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env. 模拟环境的类
            - reward_fn (:obj:`Function`): The Callable to compute the state value. 用于计算状态值的函数
            - temperature (:obj:`Int`): Temperature is a parameter that controls the "softness" of the probability distribution. 温度控制概率分布的"软化程度"
            - sample (:obj:`Bool`): The value of the node. 是否通过采样选择动作
            - return_tree (:obj:`Bool`): 是否返回决策树
        Returns:
            - action (:obj:`Bool`): Select the action with the most visits as the final action. 选择的最终动作
            - action_probs (:obj:`List`): The output probability of each action. 每个动作的概率分布
        """
        if self.root is None:
            # 如果根节点为空，初始化根节点并扩展叶节点
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, reward_fn)
            self.root = root
        else:
            root = self.root

        if root.is_leaf():
            # if root is leaf node, expand it
            # We have updated the environment legal action when we test the node is leaf node
            # So the expansion won't have bugs
            # 如果根节点是叶节点，则扩展它
            self._expand_leaf_node(root, simulate_env, reward_fn)

        if sample:
            # 如果采样，给根节点加上探索噪声
            self._add_exploration_noise(root)

        for n in range(self._num_simulations):
            # 进行多个模拟以增强策略的准确性
            simulate_env_copy = simulate_env.copy()
            simulate_env_copy.battle_mode = simulate_env_copy.mcts_mode
            self._simulate(root, simulate_env_copy, reward_fn)

        # for debugging
        # print('after simulation')
        # print('value= {}'.format([(k, v.value) for k,v in root.children.items()]))
        # print('visit_count= {}'.format([(k, v.visit_count) for k,v in root.children.items()]))

        # 收集每个动作的访问次数
        action_visits = []
        for action_dict in simulate_env.legal_actions:
            action = action_dict["action"]
            if action in root.children:
                action_visits.append((action, root.children[action].visit_count))
            else:
                action_visits.append((action, 0))

        # 计算每个动作的概率
        actions, visits = zip(*action_visits)
        action_probs = nn.functional.softmax(
            1.0
            / temperature
            * np.log(torch.as_tensor(visits, dtype=torch.float32) + 1e-10),
            dim=0,
        ).numpy()
        if sample:
            # 通过采样选择动作
            action = np.random.choice(actions, p=action_probs)
            self.reset_prior(root)
        else:
            # 选择访问次数最多的动作
            action = actions[np.argmax(action_probs)]

        # 更新根节点
        self.root = root
        if return_tree:
            return action, action_probs, root
        return action, action_probs

    def vanila_mcts(
        self,
        simulate_env: Type[CoTEnv],                 # 模拟环境类
        num_path: int,                              # 模拟的路径数量
        reward_model_fn: Optional[Callable] = None, # 奖励模型函数
        select_by_prior: bool = False,              # 是否根据先验选择动作
    ) -> List[Dict]:
        api_call_completion_tokens = 0              # 初始化API调用的token计数
        _, info = simulate_env.reset(update_legal_action=True) # 重置环境并获取初始信息
        api_call_completion_tokens += info["api_completion_token"] # 累加API token

        # 如果根节点为空，则初始化根节点并扩展
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state()) # 创建根节点
            self._expand_leaf_node(root, simulate_env, reward_model_fn) # 扩展叶节点
            self.root = root # 设置根节点

        traj_list = [] # 用于存储路径数据的列表

        # TODO(ziyu): split with 1. select 2. expand 3. rollout 4. backprop
        #  for here is split the for loop with select and rollout
        #  so that arbitrary rollout function can be used here.
        # TODO(ziyu): 将以下步骤拆分为选择、扩展、回滚和反向传播
        # 这里是将选择和回滚步骤拆开，以便可以使用任意的回滚函数
        
        for i_path in range(num_path):  # 循环生成多个路径
            node = self.root  # 从根节点开始
            env_copy = simulate_env.copy()  # 环境的副本
            done = False  # 结束标志
            while not done: # 模拟路径直到结束
                if node.visit_count > 0:
                    # if node is visited, select the child with the highest UCB score
                    # 如果节点已访问，选择UCB得分最高的子节点
                    action, node = self._select_child(node, env_copy)
                else:
                    # choose rollout policy
                    # 如果节点未被访问，根据先验概率或值选择动作
                    if select_by_prior:
                        # select with prior probability
                        # 根据先验概率选择
                        action, node = self._select_by_prior(node, env_copy)
                    else:
                        # select with highest value, since visit_count = 0 in self.ucb 
                        #  will select node with highest value
                        # 选择值最高的子节点
                        action, node = self._select_child(node, env_copy)

                # sync terminated flag here
                # XXX(ziyu): find a more clean way
                # 同步终止标志
                env_copy._next_state_terminated = {}
                assert node.last_action == action
                env_copy._next_state_terminated[action] = node.terminated

                # 执行动作并更新状态
                _, _, terminated, truncated, info = env_copy.step(
                    action, update_legal_action=node.is_leaf()
                )

                done = terminated or truncated # 如果终止或截断则结束

                # 如果节点是叶节点且未结束，扩展叶节点
                if not done and node.is_leaf():
                    self._expand_leaf_node(node, env_copy, reward_model_fn)

                # record api_tokens, if not expand, info["api_completion_token"] is 0
                # 记录API token数
                api_call_completion_tokens += info["api_completion_token"]
            else:
                # 如果节点已经被访问过，获取叶节点的值
                if node.visit_count > 0:
                    leaf_value = node.value
                else:
                    # 如果节点未被访问，使用初始值或奖励模型计算值
                    if self._init_critic_value:
                        leaf_value = node._initial_value
                    else:
                        leaf_value = reward_model_fn(env_copy.get_state()).item()
            # 更新节点的递归值
            node.update_recursive(leaf_value, env_copy.mcts_mode)

            # 记录路径数据
            traj_data = {
                "path_idx": i_path,  # 路径索引
                "text": env_copy.answer,  # 环境回答
                "value": leaf_value,  # 叶节点的值
                "api_completion_tokens": api_call_completion_tokens,  # API token的计数
                "tree_completion_tokens": self._completion_tokens,  # 树的token计数 
            }

            traj_list.append(traj_data) # 将路径数据添加到列表

            # reset api_call_completion_tokens
            # 重置API调用的token计数
            api_call_completion_tokens = 0

        return traj_list # 返回路径列表

    def beam_search(
        self,
        simulate_env: CoTEnv,
        beam_size: int,
        max_step: int,
        reward_model_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        """Beam Search implementation
        Args:
            simulate_env: The environment to simulate the search.
            beam_size: beam_size
            max_step: The maximum number of steps to search.
            reward_model_fn: The reward model function to evaluate the state.
            simulate_env: 用于模拟搜索的环境。
            beam_size: beam search 中的束大小，即每步保留的候选路径数量。
            max_step: 最大搜索步数。
            reward_model_fn: 用于评估状态的奖励模型函数。
        """
        api_call_completion_tokens = 0  # 初始化 API 调用的 token 计数
        _, info = simulate_env.reset(update_legal_action=True)  # 重置环境并获取初始信息
        api_call_completion_tokens += info["api_completion_token"]  # 累加 API token
        if self.root is None:
            # 如果根节点为空，初始化根节点并扩展
            root = LanguageNode(text_state=simulate_env.get_state())  # 创建根节点
            self._expand_leaf_node(root, simulate_env, reward_model_fn)  # 扩展叶节点
            self.root = root

        end_nodes, top_k_nodes = [], [(-root._initial_value, root, simulate_env.copy())] # 初始化结束节点和当前路径
        k = beam_size # 设定 beam size

        for _ in range(max_step + 1):  # 循环执行最多 max_step 步
            cur_nodes_to_search = top_k_nodes  # 当前待搜索的节点
            top_k_nodes = []  # 当前 step 后的候选节点列表

            # 遍历当前的路径节点，选择子节点
            for cur_neg_v, cur_node, cur_env in cur_nodes_to_search:
                if cur_node.terminated:  # 如果节点已终止，添加到结束节点列表
                    end_nodes.append((cur_neg_v, cur_node, cur_env))
                    k -= 1
                elif k > 0:  # 如果尚未达到束大小，选择子节点
                    # select at most topk children add push to heap
                    assert (
                        len(cur_node.children) > 0
                    ), "in beam search you should expand this non-terminal node at first."
                    "在 beam search 中，非终止节点需要首先扩展。"

                    # 根据值选择 top k 个子节点
                    top_k_children = sorted(
                        [
                            (action, child, child._initial_value)
                            for action, child in cur_node.children.items()
                        ],
                        key=lambda x: x[2], # 按值排序
                        reverse=True,
                    )[:k]
                    for c_act, c_node, c_value in top_k_children:
                        new_env = cur_env.copy()  # 创建环境副本
                        heapq.heappush(top_k_nodes, (-c_value, c_node, new_env))  # 按负值排序并推入堆中

            # nsmallest since we negate the value
            # 取出最小的 k 个节点
            top_k_nodes = heapq.nsmallest(k, top_k_nodes)

            # expand selected nodes
            # XXX(ziyu): this could be optimized by batch expand
            # 扩展选择的节点
            # XXX(ziyu): 可以通过批量扩展优化此部分
            for value, node, new_env in top_k_nodes:
                _, _, terminated, truncated, info = new_env.step(node.last_action, update_legal_action=True)
                api_call_completion_tokens += info["api_completion_token"]  # 累加 API token

                if terminated or truncated:  # 如果节点已终止或被截断
                    node.set_as_terminate_node()  # 设置为终止节点
                else:
                    self._expand_leaf_node(node, new_env, reward_model_fn)  # 扩展叶节点

            if len(end_nodes) == beam_size:  # 如果已经找到足够的终止节点，停止搜索
                assert k == 0
                break

        traj_list = [] # 用于存储路径数据的列表
        for i, (neg_e_v, e_node, e_env) in enumerate(end_nodes):
            traj_list.append(
                {
                    "path_idx": i,  # 路径索引
                    "text": e_env.answer,  # 环境的答案
                    "value": -neg_e_v,  # 叶节点的值（值为负数需要取负）
                    "api_completion_tokens": 0,  # API token 数量（此处为 0，最终会累加）
                    "tree_completion_tokens": 0,  # 树的 token 数量（同上）
                    # num_generated_token 难以单独计算
                    # num_generated_token is hard to compute for each single answer
                }
            )
        # 更新最后一个路径的 token 数量
        traj_list[-1]["tree_completion_tokens"] = self._completion_tokens
        traj_list[-1]["api_completion_tokens"] = api_call_completion_tokens
        return traj_list # 返回路径数据列表

    def _simulate(
        self,
        node: Node,
        simulate_env: Type[CoTEnv],
        reward_fn: Optional[Callable] = None,
    ) -> None:
        """
        Overview:
            Run a single playout from the root to the leaf, getting a value at the leaf and propagating it back through its parents.
            State is modified in-place, so a deepcopy must be provided.
            从根节点到叶节点进行一次模拟，获取叶节点的值并通过其父节点进行传播。
            状态在原地修改，因此必须提供深拷贝。
        Arguments:
            - node (:obj:`Class Node`): Current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env.
            - reward_fn (:obj:`Function`): The Callable to compute the action probs and state value.
            - node (:obj:`Class Node`): 当前节点，用于执行 MCTS 搜索。
            - simulate_env (:obj:`Class BaseGameEnv`): 模拟环境的类。
            - reward_fn (:obj:`Function`): 用于计算动作概率和状态值的函数。
        """
        # XXX: fix the bug temporally, better implementation is required.
        winner = None  # 存储赢家
        done = False  # 是否结束标志
        while not node.is_leaf():  # 如果当前节点不是叶节点，继续选择子节点
            action, node = self._select_child(node, simulate_env)  # 选择子节点
            _, _, terminated, truncated, info = simulate_env.step(
                action, update_legal_action=(node.is_leaf() and node.visit_count == 1)
            )
            done = terminated or truncated  # 检查游戏是否终止或截断

            # In original AlphaZero, the leaf node will be expanded once it is reached
            # In our setting, computing legal action is computational inefficient
            # Thus when we reach a leaf node, we will not directly expand it
            # Until the next time, when this node's children are required to be selected
            # In this case, node is leaf node and the visit count number of node is 1
            # Then we expand it
            # 在原始的 AlphaZero 中，达到叶节点时会直接扩展
            # 但在我们的实现中，计算合法动作代价较高
            # 因此，直到下次需要选择子节点时，才会扩展叶节点

            if not done and node.is_leaf() and node.visit_count == 1:
                # Once we expand the node, the node will not be leaf node any more
                # And the while won't break
                # 扩展叶节点后，节点将不再是叶节点
                self._expand_leaf_node(node, simulate_env, reward_fn)

            winner = info["winner"]  # 记录当前回合的赢家
        """
        in ``self_play_mode``, the leaf_value is calculated from the perspective of player ``simulate_env.current_player``.
        in ``play_with_bot_mode``, the leaf_value is calculated from the perspective of player 1.
        在 ``self_play_mode`` 模式下，leaf_value 从玩家 ``simulate_env.current_player`` 的角度计算。
        在 ``play_with_bot_mode`` 模式下，leaf_value 从玩家 1 的角度计算。
        """
        if not done:
            # leaf_value = self._expand_leaf_node(node, simulate_env,
            #                                     reward_fn)

            # 如果游戏未结束，计算叶节点的价值
            if not done and self.mask_non_terminal_node_value:
                leaf_value = 0.0 # 如果节点值被屏蔽，则设为0
            else:
                if not self._init_critic_value:
                    # 如果没有初始化评价值，则使用 reward_fn 计算当前状态的评价值
                    leaf_value = reward_fn(simulate_env.get_state()).item()
                else:
                    # 否则，使用初始的叶节点值
                    leaf_value = node._initial_value
        else:
            # 游戏结束时的处理
            if not self.no_terminal_reward:
                if winner is not None:
                    # 如果游戏有赢家，根据赢家记录正确或错误答案
                    if winner == 1:
                        self.answers.add(simulate_env.answer)
                    else:
                        self.wrong_answers.add(simulate_env.answer)

                # if simulate_env.mcts_mode == 'self_play_mode':
                #     if winner == -1:
                #         leaf_value = 0
                #     else:
                #         leaf_value = 1 if simulate_env.current_player == winner else -1

                if simulate_env.mcts_mode == "play_with_bot_mode":
                    # in ``play_with_bot_mode``, the leaf_value should be transformed to the perspective of player 1.
                    # 在 ``play_with_bot_mode`` 中，从玩家1的角度计算叶节点价值
                    if "reward" in info.keys():
                        leaf_value = info["reward"]
                    else:
                        if winner == -1:
                            leaf_value = 0  # 平局
                        elif winner == 1:
                            leaf_value = 1  # 玩家1胜利
                        elif winner == 2:
                            leaf_value = -1  # 玩家2胜利
            else:
                if node.visit_count > 0:
                    # because leaf value has been calculated and backpropogated
                    # 如果节点已经访问过，使用已计算的值
                    leaf_value = node.value
                else:
                    # 如果没有初始化评价值，则使用 reward_fn 计算当前状态的评价值
                    if self._init_critic_value:
                        leaf_value = node._initial_value
                    else:
                        leaf_value = reward_fn(simulate_env.get_state()).item()

        # 如果游戏结束，标记该节点为终止节点
        if done:
            node.set_as_terminate_node()
            if self.visited_paths is not None:
                self.visited_paths.append(
                    {
                        "text": simulate_env.answer,
                        "correct": winner == 1,  # 判断答案是否正确
                        "value": leaf_value,  # 记录叶节点的价值
                    }
                )

        # Update value and visit count of nodes in this traversal.
        # 更新遍历中节点的值和访问次数
        if simulate_env.mcts_mode == "play_with_bot_mode":
            # 在与机器人对战模式下，更新节点的值
            node.update_recursive(leaf_value, simulate_env.mcts_mode)

        elif simulate_env.mcts_mode == "self_play_mode":
            # NOTE: e.g.
            #       to_play: 1  ---------->  2  ---------->  1  ----------> 2
            #         state: s1 ---------->  s2 ---------->  s3 ----------> s4
            #                                     action    node
            #                                            leaf_value
            # leaf_value is calculated from the perspective of player 1, leaf_value = value_func(s3),
            # but node.value should be the value of E[q(s2, action)], i.e. calculated from the perspective of player 2.
            # thus we add the negative when call update_recursive().
            # 在自对弈模式下，计算叶节点的值时需要加上负号
            # 因为叶节点的值是从玩家1的角度计算的，但我们需要更新的是玩家2的角度
            node.update_recursive(-leaf_value, simulate_env.mcts_mode)

    def _select_child(
        self, node: LanguageNode, simulate_env: Type[CoTEnv]
    ) -> Tuple[Union[int, float], Node]:
        """
        Overview:
            Select the child with the highest UCB score.
            选择具有最高 UCB（上置信界）分数的子节点
        Arguments:
            - node (:obj:`Class Node`): Current node.
            - node (:obj:`Class Node`): 当前节点
        Returns:
            - action (:obj:`Int`): choose the action with the highest ucb score.
            - child (:obj:`Node`): the child node reached by executing the action with the highest ucb score.
             - action (:obj:`Int`): 选择具有最高 UCB 分数的动作
             - child (:obj:`Node`): 通过执行最高 UCB 分数的动作到达的子节点
        """

        action = None
        child = None
        best_score = -9999999  # 初始化最好的分数为一个极小值

        # 遍历所有子节点，计算 UCB 分数
        for action_tmp, child_tmp in node.children.items():
            ucb_score = self._ucb_score(node, child_tmp)  # 计算当前子节点的 UCB 分数
            score = ucb_score
            if score > best_score:
                best_score = score  # 更新最好的分数
                action = action_tmp  # 更新选择的动作
                child = child_tmp  # 更新选择的子节点

        # 如果没有找到合适的子节点，则返回当前节点（可能是叶节点）
        if child is None:
            child = node  # child==None, node is leaf node in play_with_bot_mode.   # 如果没有子节点，说明当前节点是叶节点

        return action, child  # 返回选择的动作和子节点

    def _select_by_prior(self, node: Node, simulate_env):
        """
        根据先验概率选择子节点。
        参数：
            - node (:obj:`Class Node`): 当前节点。
            - simulate_env: 当前环境（未使用）。
        返回：
            - chosen_action (:obj:`Int`): 根据先验概率选择的动作。
            - chosen_node (:obj:`Node`): 选择的子节点。
        """

        # 根据每个子节点的先验概率构建一个列表
        data_tmp = [
            (x_action, x_node.prior_p) for x_action, x_node in node.children.items()
        ]
        action_list, prior_list = list(zip(*data_tmp)) # 提取动作列表和对应的先验概率列表

        # 根据先验概率随机选择一个动作
        chosen_action = np.random.choice(action_list, p=np.array(prior_list))
        chosen_node = node.children[chosen_action] # 获取对应的子节点

        return chosen_action, chosen_node # 返回选择的动作和子节点

    def _expand_leaf_node(
        self,
        node: Node,
        simulate_env: Type[CoTEnv],
        reward_fn: Optional[Callable] = None,
    ) -> float:
        """
        Overview:
            expand the node with the reward_fn.
            扩展叶子节点并计算奖励值
        Arguments:
            - node (:obj:`Class Node`): current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): the class of simulate env.
            - reward_fn (:obj:`Function`): the Callable to compute the state value.
            - node (:obj:`Class Node`): 当前节点，在执行 MCTS 搜索时使用。
            - simulate_env (:obj:`Class BaseGameEnv`): 模拟环境的类。
            - reward_fn (:obj:`Function`): 用于计算状态值的可调用函数。
        Returns:
            - leaf_value (:obj:`Bool`): the leaf node's value.
            - leaf_value (:obj:`Float`): 叶子节点的值（奖励值）
        """
        """
        action_probs_dict, leaf_value = reward_fn(simulate_env)
        for action, prior_p in action_probs_dict.items():
            if action in simulate_env.legal_actions:
                node.children[action] = Node(parent=node, prior_p=prior_p)
        """

        # 获取模拟环境的当前状态
        text_state = simulate_env.get_state()
        # 如果没有初始化评论器值，直接使用奖励函数计算叶子节点的值
        if not self._init_critic_value:
            leaf_value = reward_fn(text_state)

        else:
            # 如果已初始化评论器值，则使用节点的初始值
            leaf_value = node._initial_value
            assert len(simulate_env.legal_actions) > 0 # 确保有合法动作可供选择

            # 使用奖励函数计算每个合法动作的预期奖励（PRM）
            prms = reward_fn(
                [
                    (
                        simulate_env.question,
                        simulate_env.answer + x["action"], # 合并当前的动作
                    )
                    for x in simulate_env.legal_actions # 会调用update_action函数生成leagl_actions
                ]
            )
            # 存储每个动作的值
            child_values = []
            # PRM get last r as single reward
            # 处理每个动作的PRM值（只取最后一个值作为奖励）
            for act, rs in zip(simulate_env.legal_actions, prms):
                if len(simulate_env.action_history) + 1 != len(rs):
                    logger.warning(
                        # PRM值长度与动作历史不匹配
                        "PRM value length not match with action history. \
                            len(prm)={}, len(act_hist)={} s:\n {}\n\na: \n{}\nrs:{}".format(
                            len(prms),
                            len(simulate_env.action_history),
                            text_state,
                            act,
                            rs,
                        )
                    )
                    # raise RuntimeError("Tokenizer problems")
                    # 如果长度不匹配，设置子节点值为0
                    child_values.append(0.0)

                # TODO: quick fix of extra reward labeling in tree expand_leaf_node (#77) on Dec 7, 2024
                #if len(rs) == 0:
                elif len(rs) == 0:
                    logger.warning(
                        # PRM值为空
                        "Empty PRM value for: \nState: \n{} \naction: \n{}, will be set to 0.0".format(
                            text_state, act
                        )
                    )
                    # 如果PRM值为空，设置子节点值为0
                    child_values.append(0.0)
                else:
                    # prm-last
                    # 使用PRM的最后一个值作为奖励
                    child_values.append(rs[-1])
                    # # prm-min
                    # 也可以选择取PRM中的最小值
                    # child_values.append(min(rs))
                    # # prob-prm
                    # 或者使用动作的概率值
                    # child_values.append(act['prob'])

        # 确保当前节点没有子节点
        assert len(node.children) == 0
        # 遍历所有合法动作
        for i, action_dict in enumerate(simulate_env.legal_actions):
            action, prob = action_dict["action"], action_dict["prob"]

            # 如果已初始化评论器值，则使用对应的子节点值
            if self._init_critic_value:
                child_value = child_values[i]
            else:
                # XXX(ziyu): consider turn off this branch, i.e. always assume
                #  `self._init_critic=True`, since with LLM
                # 否则，默认为 0.0
                child_value = 0.0

            # 为当前动作创建子节点
            node.children[action] = LanguageNode(
                parent=node,
                prior_p=prob,               # 设置子节点的优先概率
                #  prm_value=prm_value,
                text_state=text_state,      # 当前状态
                last_action=action,         # 当前动作
                initial_value=child_value,  # 初始值
                num_generated_token=action_dict["num_token"], # 生成的 token 数量
            )
            # set terminal node here
            # 如果当前动作导致状态终止，则将其标记为终止节点
            if simulate_env._next_state_terminated[action]:
                node.children[action].set_as_terminate_node()
        # 如果没有生成任何子节点，打印调试信息
        if len(node.children) == 0:
            print_rank_0(
                "Prune all current children at node {}".format(node.last_action)
            )
        
        # collect num tokens
        # 收集子节点生成的 token 数量
        if not node.has_collected_token_num:
            self._completion_tokens += sum(
                c.num_generated_token for c in node.children.values()
            )
            node.has_collected_token_num = True
        else:
            # 如果已经收集过 token 数量，抛出异常
            raise RuntimeError("Token number has been collected again.")

        return leaf_value # 返回叶子节点的值

    def _ucb_score(self, parent: Node, child: Node) -> float:
        """
        Overview:
            Compute UCB score. The score for a node is based on its value, plus an exploration bonus based on the prior.
            计算 UCB (Upper Confidence Bound) 分数。该分数基于节点的值，并加上一个基于先验概率的探索奖励
        Arguments:
            - parent (:obj:`Class Node`): Current node.
            - child (:obj:`Class Node`): Current node's child.
            - parent (:obj:`Class Node`): 当前节点
            - child (:obj:`Class Node`): 当前节点的子节点
        Returns:
            - score (:obj:`Bool`): The UCB score.
            - score (:obj:`float`): 计算得到的 UCB 分数
        """
        # 计算探索常数 pb_c
        pb_c = (
            math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base)
            + self._pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        # 先验得分 (prior_score) 和当前节点值得分 (value_score)
        prior_score = pb_c * child.prior_p
        value_score = child.value

        # 返回总的 UCB 分数
        return prior_score + value_score
        # return value_score

    def reset_prior(self, node: Node) -> None:
        """
        Overview:
            Reset prior probability
            重置先验概率
        Arguments:
            - node (:obj:`Class Node`): Current node.
            - node (:obj:`Class Node`): 当前节点
        """
        # 遍历当前节点的所有子节点，重置每个子节点的先验概率
        for a in node.children.keys():
            node.children[a].prior_p = node.children[a].prior_p_ori # 恢复原始先验概率

    def _add_exploration_noise(self, node: Node) -> None:
        """
        Overview:
            Add exploration noise.
            向当前节点添加探索噪声
        Arguments:
            - node (:obj:`Class Node`): Current node.
            - node (:obj:`Class Node`): 当前节点
        """
        # Get a list of actions corresponding to the child nodes.
        # 获取子节点对应的动作列表
        actions = list(node.children.keys())
        # Create a list of alpha values for Dirichlet noise.
        # 创建 Dirichlet 噪声的 alpha 参数列表
        alpha = [self._root_dirichlet_alpha] * len(actions)
        # Generate Dirichlet noise using the alpha values.
        # 使用 alpha 值生成 Dirichlet 噪声
        noise = np.random.dirichlet(alpha)
        # Compute the weight of the exploration noise.
        # 设置探索噪声的权重比例
        frac = self._root_noise_weight
        # Update the prior probability of each child node with the exploration noise.
        # 更新每个子节点的先验概率，加入探索噪声
        for a, n in zip(actions, noise):
            node.children[a].prior_p = node.children[a].prior_p * (1 - frac) + n * frac

    @classmethod
    def from_json(cls, cfg: dict, json_path: str, reset_visit_info: bool):
        """
        概述:
            从 JSON 文件加载并构建树形结构。
        参数:
            - cfg (:obj:`dict`): 配置字典。
            - json_path (:obj:`str`): JSON 文件路径。
            - reset_visit_info (:obj:`bool`): 是否重置访问信息。
        返回:
            - obj (:obj:`Class`): 构建好的树结构对象。
        """
        # 读取 JSON 文件内容
        tree_json = json.load(open(json_path, "r"))

        def build_tree(tree_dict: dict) -> Node:
            """
            构建树的递归函数。
            """
            # 获取节点信息
            node_info = tree_dict["info"]
            # 创建当前节点
            current_node = LanguageNode(
                text_state=node_info.get("text_state", None),
                last_action=node_info.get("last_action", None),
                prior_p=node_info["prior_p"],
                prm_value=node_info.get("prm_value", None),
                initial_value=node_info.get("initial_value", 0.0),
            )

            # 如果不重置访问信息，加载访问次数和累积值
            if not reset_visit_info:
                current_node._visit_count = node_info["visit_cnt"]
                current_node._value_sum = node_info["value"] * current_node.visit_count
            # 如果节点已终止，设置为终止节点
            if node_info.get("terminated", False):
                current_node.set_as_terminate_node()

            # 递归构建子节点
            for name, child_dict in tree_dict["children"].items():
                child_node = build_tree(child_dict)
                current_node._children[name] = child_node
                child_node._parent = current_node

            return current_node

        # 构建树的根节点
        root_node = build_tree(tree_dict=tree_json)

        # 创建类实例并设置根节点
        obj = cls(cfg)
        obj.root = root_node
        return obj
    
    def draw_tree(self):
        """
        概述:
            打印树的结构，用于可视化展示。
        """
        # Not tested yet
        root = self.root
        assert root, 'Root node is None'
        def draw_node(node, depth):
            """
            递归绘制树的每个节点。
            """
            print('|' + '-' * depth + str(node))
            for child in node.children.values():
                draw_node(child, depth + 1)
        
        print(f"\n---------Expanded Tree---------")
        draw_node(self.root)
