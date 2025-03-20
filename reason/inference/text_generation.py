from typing import List, Optional
import requests
from dataclasses import dataclass


# 定义一个数据类，用于存储语言模型生成的结果
@dataclass
class ConcatedLMGenResult:
    text: List[str]                  # 生成的文本列表
    prompt_tokens: List[int]         # 提示文本的token数量
    num_tokens: List[int]            # 生成文本的token数量
    cumulative_logprob: List[float]  # 累计对数概率列表
    logp_avg_by_len: List[float]     # 按长度加权的平均对数概率
    finish_reason: List[str]         # 生成结束的原因（如停止标志）

    # post init compute number of completion_tokens
    # 初始化后计算生成文本的token总数
    def __post_init__(self):
        self.completion_tokens = sum(self.num_tokens)  # 计算生成文本的token总数


# 定义生成文本的函数，向语言模型服务发送请求并返回生成结果
def _generate_fastchat(
    query_str,                  # 输入的提示文本
    model_name,                 # 使用的模型名称
    n,                          # 生成样本的数量
    temperature,                # 控制随机性的温度值
    top_p,                      # nucleus sampling的p值
    top_k,                      # top-k采样的k值
    max_new_tokens,             # 生成的最大token数量
    stop_token_ids,             # 停止生成的token ID列表
    stop_str,                   # 停止生成的字符串
    include_stop_str_in_output, # 是否将停止字符串包含在输出中
    controller_addr,            # 控制器的地址
) -> ConcatedLMGenResult:
    
    # 获取工作节点地址（与指定模型对应的服务节点）
    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    
    # 如果没有找到有效的工作节点地址，抛出错误
    if not worker_addr:
        raise ValueError(f"Language Model name {model_name} does not exist.")
    
    headers = {"User-Agent": "FastChat Client"}  # 请求头
    
    # 构建生成请求的参数
    gen_params = {
        "model": model_name,            # 模型名称
        "prompt": query_str,            # 输入的提示文本
        "temperature": temperature,     # 随机性温度
        "n": n,                         # 生成样本数量
        "top_p": top_p,                 # nucleus sampling的p值
        "top_k": top_k,                 # top-k采样的k值
        "stop_token_ids": stop_token_ids,  # 停止生成的token ID列表
        "max_new_tokens": max_new_tokens,  # 最大生成token数量
        "stop": stop_str,               # 停止生成的字符串
        "echo": False,                  # 不回显输入
        "include_stop_str_in_output": include_stop_str_in_output,  # 是否在输出中包括停止字符串
    }
    
    # 向工作节点请求生成文本，使用流式响应
    response = requests.post(
        worker_addr + "/worker_generate",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    
    # 解析生成结果
    results = response.json()
    
    # 提取输出文本的token长度和累计对数概率
    output_token_lens = results["output_token_len"]
    cum_logps = results["cumulative_logprob"]
    
    # 计算每个生成文本的平均对数概率（按文本长度加权）
    avg_len_logps = [
        clp / max(1, otl) for clp, otl in zip(cum_logps, output_token_lens)
    ]
    # return results["text"], avg_len_logps
    # 返回封装好的生成结果
    return ConcatedLMGenResult(
        text=results["text"],  # 生成的文本
        prompt_tokens=results["usage"]["prompt_tokens"],  # 提示文本的token数量
        num_tokens=results["output_token_len"],           # 生成文本的token数量
        cumulative_logprob=cum_logps,                     # 累计对数概率
        logp_avg_by_len=avg_len_logps,                    # 按长度加权的平均对数概率
        finish_reason=results["finish_reason"],           # 生成结束的原因
    )
