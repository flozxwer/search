from collections import Counter, defaultdict
from typing import List

# 常量定义：聚合方法的标识
MAJORITY_VOTE = "majority_vote"
ORM_VOTE = "orm_vote"
ORM_MAX = "orm_max"
PRM_MIN_MAX = "prm_min_max"
PRM_MIN_VOTE = "prm_min_vote"
PRM_LAST_MAX = "prm_last_max"
PRM_LAST_VOTE = "prm_last_vote"

# _代表私有方法
# 主要聚合方法：
# 处理输入的 x_list 和 v_list 数据

# 投票法，选择出现次数最多的元素
def _agg_majority_vote(x_list: List[str], unused_v_list: List[float]):
    counts = Counter(x_list)  # 统计每个元素的出现次数
    most_common = max(counts, key=counts.get)  # 选择出现最多的元素
    return most_common

# ORM 投票法：基于加权投票，选择得分最高的元素
def _agg_orm_vote(x_list: List[str], v_list: List[float]):
    assert len(x_list) == len(v_list)  # 确保输入的元素和权重列表长度一致
    x_dict = defaultdict(lambda: 0.0)  # 默认值为 0.0 的字典
    for x, v in zip(x_list, v_list):  # 遍历元素与其对应的权重
        x_dict[x] += v  # 累加对应元素的权重

    highest_x = max(x_dict, key=x_dict.get)  # 选择得分最高的元素
    return highest_x


# ORM 最大值法：选择对应最大权重的元素
def _agg_orm_max(x_list: List[str], v_list: List[float]):
    text_max = x_list[v_list.index(max(v_list))]  # 选择对应最大值的元素
    return text_max

# PRM 最小值最大法：首先对每个子列表取最小值，再选择最大值对应的元素
def _agg_prm_min_max(x_list: List[str], v_list: List[List[float]]):
    v_list = [min(v) if v else -1.0 for v in v_list]  # 对每个子列表取最小值，空列表视为 -1.0
    text_max = x_list[v_list.index(max(v_list))]  # 选择对应最大值的元素
    return text_max

# PRM 最后值最大法：选择每个子列表最后一个值，取最大值对应的元素
def _agg_prm_last_max(x_list: List[str], v_list: List[List[float]]):
    v_list = [v[-1] if v else -1.0 for v in v_list]  # 获取每个子列表的最后一个值，空列表视为 -1.0
    text_max = x_list[v_list.index(max(v_list))]  # 选择对应最大值的元素
    return text_max

# PRM 最小值投票法：对每个子列表取最小值，再进行加权投票选择元素
def _agg_prm_min_vote(x_list: List[str], v_list: List[List[float]]):
    v_list = [min(v) if v else -1.0 for v in v_list]  # 对每个子列表取最小值，空列表视为 -1.0
    return _agg_orm_vote(x_list, v_list)  # 使用 ORM 投票法选择元素


# PRM 最后值投票法：选择每个子列表最后一个值，再进行加权投票选择元素
def _agg_prm_last_vote(x_list: List[str], v_list: List[List[float]]):
    v_list = [v[-1] if v else -1.0 for v in v_list]  # 获取每个子列表的最后一个值，空列表视为 -1.0
    return _agg_orm_vote(x_list, v_list)  # 使用 ORM 投票法选择元素

# 聚合函数映射表：根据方法名称选择对应的聚合函数
AGG_FN_MAP = {
    MAJORITY_VOTE: _agg_majority_vote,   # 投票法
    # ORM_VOTE: _agg_orm_vote,           # 加权投票法
    # ORM_MAX: _agg_orm_max,             # 最大值法
    PRM_MIN_MAX: _agg_prm_min_max,       # PRM 最小值最大法
    PRM_MIN_VOTE: _agg_prm_min_vote,     # PRM 最小值投票法
    PRM_LAST_MAX: _agg_prm_last_max,     # PRM 最后值最大法
    PRM_LAST_VOTE: _agg_prm_last_vote,   # PRM 最后值投票法
}