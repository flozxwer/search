"""
rStar Implementation
"""

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
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict

from .tree import Node, LanguageNode, SearchTree
from envs.rstar.rstar_utils import *


class RstarSearchTree(SearchTree):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.parent2children: Dict[RstarLanguageNode, List[RstarLanguageNode]] = dict()  # 存储每个节点的子节点
        self.explored_nodes = set()                            # 已探索的节点集合
        self.N: Dict[RstarLanguageNode, int] = defaultdict(
            lambda: 0
        )  # total visit count for each node                   # 每个节点的访问次数
        self.Q: Dict[MCTS_Node, float] = defaultdict(
            lambda: 0.0
        )  # total reward of each node                         # 每个节点的奖励总和
        self.weight_scheduler = "const"                        # 权重调度策略
        self.mcts_exploration_weight = 2.0                     # 探索时的权重
        self.max_depth_allowed = 5                             # 允许的最大深度
        self.show_tree_expansion = True                        # 是否展示树的扩展

    @override
    def _select_child(
        self, node: RstarLanguageNode, simulate_env: Type[CoTEnv], rollout_id: int
    ) -> Tuple[RstarLanguageNode, bool]:

        # for select, if there is unexplored children, select it randomly. if all children nodes
        # have been explored, select UCB, a leaf node means it has no children, return True
        # when there is unexplored node

        # 如果当前节点没有子节点，返回当前节点
        if node not in self.parent2children.keys():
            return node, True

        # if there are children unexplored
        # 如果还有未探索的子节点，随机选择一个
        unexplored = [
            n for n in self.parent2children[node] if n not in self.explored_nodes
        ]
        if unexplored:
            next_node = random.choice(unexplored)
            return next_node, True

        # if all have been explord, from parent2children dict, select one node with highest UCB score

        # Get the list of children for the current node
        # 如果所有子节点都已探索，使用UCB选择最优子节点
        children = self.parent2children[node]

        # Compute UCT values for each child node
        uct_values = {
            n: self._compute_uct(parent_node=node, node=n, rollout_id=rollout_id)
            for n in children
        }
        # print(f"@@@ uct = {uct_values}, node type = {[i.node_type for i in children]}")
        # Find the child with the maximum UCT value
        # 返回具有最大UCT值的子节点
        next_node = max(uct_values, key=uct_values.get)

        return next_node, False

    def _compute_uct(
        self, parent_node: RstarLanguageNode, node: RstarLanguageNode, rollout_id: int
    ):
        "Upper confidence bound for trees"
        "计算UCT值（上置信界）"
        if parent_node is None:  # invalid UCT: the node is the root # 根节点没有UCT值
            return 666
        else:
            if self.N[node] == 0:  # invalid UCT: the node has not been explored yet # 如果节点未被探索，返回一个较大的值
                return 999
            else:
                weight = self._get_weight(rollout_id) # 获取当前探索权重
                # 计算UCT值，包含奖励/访问次数和探索因子
                return self.Q[node] / self.N[node] + weight * math.sqrt(
                    math.log(self.N[parent_node]) / self.N[node]
                )

    def _get_weight(self, rollout_id: int):
        # start with exploration weight, end with 0.1 * exploration weight
        # 根据不同策略调整探索权重
        if self.weight_scheduler == "exp":
            return self.mcts_exploration_weight * (0.1 ** (rollout_id / self.num_path)) # 指数衰减
        elif self.weight_scheduler == "lin":
            return self.mcts_exploration_weight * (
                1 - 0.9 * (rollout_id / self.num_path)   # 线性衰减
            )
        elif self.weight_scheduler == "const":
            return self.mcts_exploration_weight   # 常数权重

    def rstar_mcts(
        self,
        simulate_env: Type[CoTEnv],                 # 模拟环境
        num_path: int,                              # 路径数量
        reward_model_fn: Optional[Callable] = None, # 奖励模型函数（可选）
        select_by_prior: bool = False,              # 是否根据先验选择
    ) -> List[Dict]:  # 返回路径的列表
        simulate_env.reset()  # 重置环境
        # api_call_completion_tokens += info["api_completion_token"]
        if self.root is None:
            # 如果根节点为空，初始化根节点
            root = RstarLanguageNode(
                id=0,
                parent=None,
                depth=0,
                node_type=Node_Type.USER_QUESTION,
                disable_a5=False,
                user_question=simulate_env.math_problem["question"], # 获取问题
                expected_answer=simulate_env.math_problem["answer"], # 获取答案
                max_depth_allowed=self.max_depth_allowed, # 最大允许深度
                disable_a1=False,
            )
            self.root = root # 设置根节点

        traj_list = []              # 存储路径信息
        self.num_path = num_path    # 设置路径数量
        model_solutions = []        # 存储模型解
        model_all_solutions = []    # 存储所有解
        model_rollout_nodes = []    # 存储回合路径中的节点

        # 执行 MCTS 模拟 num_path 次
        for i_path in tqdm(range(num_path), desc=f"Running {num_path} MCTS paths"):
            node = self.root  # 从根节点开始
            env_copy = simulate_env.copy() # 复制环境
            done = False # 终止条件
            node_path = []  # for boostrapping # 存储路径上的节点，便于后期回溯更新
            # 在当前路径中进行遍历，直到到达终止节点
            while not done:
                # this is the whole process of navigating from root to a terminate node, along the way,
                # there are explored nodes with children where we do UCB, and there are unexplored nodes where we
                # expand its children through legal_action_update,
                # for select, if there is unexplored children, select it randomly. if all children nodes
                # have been explored, select UCB
                # 这是从根节点到终止节点的整个过程，在此过程中：
                # 有一些已探索的节点和它们的子节点，在这些节点上我们使用 UCB（上置信界）算法。
                # 有一些未探索的节点，我们通过 `legal_action_update` 扩展其子节点。
                # 对于选择操作，如果存在未探索的子节点，则随机选择一个。如果所有子节点都已被探索，则使用 UCB 选择。
                # 从当前节点选择一个子节点
                next_node, is_leaf = self._select_child(
                    node, env_copy, i_path
                )  # find a leaf node or
                # simulate the remaining  # 查找一个叶子节点或继续扩展
                node_path.append(next_node) # 将选择的节点加入路径

                done = env_copy.is_terminal(next_node)  # checking terminal condition # 判断是否为终止节点
                # update legal action (expand) when the current code is not a leaf (no children)
                # no env.step only use for checking termination when is not leaf, and update legal action
                # when the current node is leaf
                # print(f"Path {i_path}: depth = {next_node.depth}, done = {done}, is leaf = {is_leaf}")

                # 如果当前节点是叶节点，尝试扩展其子节点
                if (
                    not done and is_leaf
                ):  # expand when encounter a leaf node one step further
                    next_node_children = env_copy.try_update_legal_action(
                        node=next_node
                    )
                    for c in next_node_children:
                        c.set_rollout_id(i_path) # 设置回合 ID
                    self.parent2children[next_node] = next_node_children # 更新子节点

                if self.show_tree_expansion:
                    self.draw_tree() # 可视化树的扩展

                if done:
                    self.explored_nodes.add(next_node) # 将终止节点标记为已探索

                node = next_node # 更新当前节点

            else:
                # boostrapping
                # 回溯时更新节点的 Q 值和访问次数
                reward = next_node.calculate_reward() # 计算奖励

                for node in reversed(node_path): # 反向遍历路径节点
                    self.Q[node] += reward  # 更新 Q 值
                    self.N[node] += 1 # 更新访问次数
                    self.explored_nodes.add(node) # 将节点标记为已探索

            model_rollout_nodes.append(next_node) # 将当前路径的最后一个节点加入回合路径
            # 查找最佳解并记录路径信息
            _, best_solution, _, chosen_node, all_solution_nodes, all_solutions = (
                stochastic_find_best_solution(
                    self.root, env_copy.evaluator, enable_potential_score=False
                )
            )

            # model_solutions.append(best_solution)
            # model_all_solutions.append(all_solutions)
            # 确保最佳解存在
            assert best_solution is not None

            # 存储当前路径的解和相关信息
            traj_data = {
                "path_idx": i_path,        # 路径索引
                "text": best_solution,     # 最佳解
                "value": reward,           # 奖励值
                "api_completion_tokens": env_copy.  total_api_call_completion,   # API 调用的 token 数
                "tree_completion_tokens": env_copy.total_tree_completion, # 树的 completion token 数
            }

            traj_list.append(traj_data) # 将当前路径数据加入结果列表

        return traj_list # 返回所有路径的列表

    def draw_tree(self):
        # 定义递归函数来显示树的节点
        def display_tree(node):
            # 打印当前节点，节点深度决定缩进
            print("|" + "-" * (node.depth * 4) + str(node))
            # 递归打印当前节点的所有子节点
            for child in node.children:
                display_tree(child)

        # 打印树的起始信息
        print(f"\n---------Expanded Tree---------")
        # 从根节点开始显示整个树
        display_tree(self.root)
