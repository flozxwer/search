import torch
from typing import Union, List
from transformers import AutoTokenizer
import re
import numpy as np
import requests


def _value_inference_fastchat(
    model_name: str,                   # 模型名称
    input_str: Union[List[str], str],   # 输入文本，可以是字符串或字符串列表
    controller_addr="http://0.0.0.0:28777",  # 控制器地址
):
    # 获取与指定模型对应的工作节点地址
    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    
    # 如果没有找到有效的工作节点地址，抛出错误
    if not worker_addr:
        raise ValueError("Value Model name {} does not exist.".format(model_name))

    headers = {"User-Agent": "FastChat Client"}  # 请求头
    
    # 构建推理请求的参数
    gen_params = {"input_str": input_str}
    
    # 向工作节点发送推理请求
    response = requests.post(
        worker_addr + "/worker_value_inference",  # 推理服务的地址
        headers=headers,
        json=gen_params,
        stream=True,
    )
    
    # 解析并提取推理结果中的值
    results = response.json()
    value = results["value"]
    
    return value  # 返回推理结果
