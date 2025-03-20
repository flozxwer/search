from importlib import import_module
from transformers import PreTrainedTokenizer

# **kwargs：关键字参数（通常是一个字典），以 key=value 的形式传入的参数

# 获取指定环境的训练和测试数据集
def get_env_datasets(env_name: str, **kwargs):
    task_module = import_module(f"envs.{env_name}")  # 动态导入指定的环境模块
    return task_module.get_train_test_dataset(**kwargs)  # data.py

# 获取默认的查询字符串构建函数
def get_default_query_str_builder(env_name: str, **kwargs):
    task_module = import_module(f"envs.{env_name}")  # 动态导入指定的环境模块

    def fn(problem_input: str, is_few_shot: bool):
        # 使用环境模块中的方法构建查询字符串
        return task_module.Env.build_query_str(
            cot_task_desc=task_module.COT_TASK_DESC,            # 任务描述
            cot_examples=task_module.COT_EXAMPLES,              # 任务示例
            problem_format_str=task_module.PROBLEM_FORMAT_STR,  # 问题格式化字符串
            problem_input=problem_input,                        # 问题输入
            is_few_shot=is_few_shot,                            # 是否为少样本任务
        )

    return fn  # 返回查询字符串构建函数

# 获取默认的响应字符串构建函数
def get_default_response_str_builder(env_name: str, **kwargs):
    task_module = import_module(f"tsllm.envs.{env_name}")  # 动态导入指定的环境模块

    def fn(problem_input: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool):
        # 使用环境模块中的方法构建响应字符串
        return task_module.Env.build_response_str(
            problem_input,  # 问题输入
            tokenizer,  # 分词器
            add_eos_token,  # 是否添加结束符
        )

    return fn  # 返回响应字符串构建函数

# 获取指定环境的答案检查器
def get_env_answer_checker(env_name):
    task_module = import_module(f"envs.{env_name}")  # 动态导入指定的环境模块

    def judge_answer(problem_str, groundtruth_str, answer_completion: str):
        # 使用环境模块中的方法判断答案是否正确
        return task_module.judge_correct(
            problem_str,  # 问题字符串
            task_module.extract_groundtruth(groundtruth_str),  # 提取真实答案
            task_module.extract_answer(answer_completion),  # 提取生成的答案
        )

    return judge_answer  # 返回答案判断函数

