from dataclasses import dataclass
from typing import List, Optional, Union
from reason.inference.text_generation import ConcatedLMGenResult, _generate_fastchat


@dataclass
class LMCallingConfig:
    """
    配置语言模型调用的参数。
    """
    n: int = 1                                  # 生成的样本数量
    temperature: float = 1.0                    # 温度控制，影响生成文本的随机性
    top_p: float = 1.0                          # nucleus采样的p值
    top_k: int = -1                             # top-k采样，-1表示禁用
    max_new_tokens: int = 512                   # 最大生成的token数
    stop_token_ids: Optional[List[int]] = None  # 停止生成的token IDs
    stop_str: Optional[Union[str, List[str]]] = None  # 停止符字符串，可以是单个字符串或多个字符串
    include_stop_str_in_output: bool = False    # 是否将停止符包括在输出中


class LanguageModelCallingFunction:
    """
    语言模型调用的基类，定义了调用接口。
    """
    def __init__(self, lm_step_tag: str = None):
        self.lm_step_tag = lm_step_tag  # 可选的模型步骤标签，用于标识调用步骤

    def __call__(self, input_str: str, config: LMCallingConfig) -> 'ConcatedLMGenResult':
        """
        子类需要实现此方法来处理语言模型生成。
        """
        raise NotImplementedError


class VLLMRemoteCaller(LanguageModelCallingFunction):
    """
    远程调用VLLM模型的实现类。
    """
    def __init__(
        self,
        model_name: str,
        controller_addr: str = "http://0.0.0.0:28777",  # 默认的控制器地址
        lm_step_tag: str = None,
    ):
        self.model_name = model_name  # 模型名称
        self.controller_addr = controller_addr  # 控制器地址
        super().__init__(lm_step_tag)

    def __call__(self, input_str: str, config: LMCallingConfig) -> 'ConcatedLMGenResult':
        """
        调用远程VLLM模型生成结果。
        """
        return _generate_fastchat(
            query_str=input_str,                    # 输入的查询字符串
            model_name=self.model_name,             # 使用的模型名称
            n=config.n,                             # 生成的样本数量
            temperature=config.temperature,         # 温度参数
            top_p=config.top_p,                     # nucleus采样的p值
            top_k=config.top_k,                     # top-k采样
            max_new_tokens=config.max_new_tokens,   # 最大生成token数
            stop_token_ids=config.stop_token_ids,   # 停止token的ID列表
            stop_str=config.stop_str,               # 停止符
            controller_addr=self.controller_addr,   # 控制器地址
            include_stop_str_in_output=config.include_stop_str_in_output, # 是否将停止符包括在输出中
        )


class FastChatRemoteCaller(LanguageModelCallingFunction):
    def __init__(
        self,
        model_name,
        controller_addr="http://0.0.0.0:28777",  # 默认控制器地址
        lm_step_tag: str = None,  # 可选的模型步骤标签
    ):
        self.model_name = model_name  # 模型名称
        self.controller_addr = controller_addr  # 控制器地址
        super().__init__(lm_step_tag)  # 调用父类初始化方法

    def __call__(self, input_str: str, config: LMCallingConfig) -> ConcatedLMGenResult:
        """
        调用远程FastChat模型生成结果（低效实现，不能处理大量调用）。
        """
        # XXX(ziyu): Low-efficiency implementation, can not accept to much calls
        # 存储每次生成的结果
        text = []                # 生成的文本
        prompt_token = []        # 提示token数量
        num_tokens = []          # 生成的token数量
        cumulative_logprob = []  # 累计对数概率
        logp_avg_by_len = []     # 按长度平均的对数概率
        finish_reason = []       # 生成结束的原因

        # 循环执行生成，config.n表示生成n个样本
        for i in range(config.n):
            # 调用 _generate_fastchat 生成单个样本
            res = _generate_fastchat(
                query_str=input_str,                   # 输入字符串
                model_name=self.model_name,            # 使用的模型名称
                n=1,                                   # 生成1个样本（此参数不使用）
                temperature=config.temperature,        # 生成的温度值
                top_p=config.top_p,                    # nucleus采样的p值
                top_k=config.top_k,                    # top-k采样的k值
                max_new_tokens=config.max_new_tokens,  # 最大生成的token数
                stop_token_ids=config.stop_token_ids,  # 停止token的ID列表
                stop_str=config.stop_str,              # 停止符
                controller_addr=self.controller_addr,  # 控制器地址
                include_stop_str_in_output=config.include_stop_str_in_output,            # 是否包括停止符在输出中
            )
            # 收集生成的结果
            text.append(res.text[0])  # 生成的文本
            cumulative_logprob.append(res.cumulative_logprob[0])  # 累计对数概率
            logp_avg_by_len.append(res.logp_avg_by_len[0])  # 按长度平均的对数概率
            prompt_token.append(res.prompt_tokens[0])  # 提示token数量
            num_tokens.append(res.num_tokens[0])  # 生成的token数量
            finish_reason.append(res.finish_reason[0])  # 生成结束的原因

        # 返回一个包含所有生成结果的对象
        return ConcatedLMGenResult(
            text=text,  # 生成的文本列表
            prompt_tokens=prompt_token,  # 提示token数量列表
            num_tokens=num_tokens,  # 生成的token数量列表
            cumulative_logprob=cumulative_logprob,  # 累计对数概率列表
            logp_avg_by_len=logp_avg_by_len,  # 按长度平均的对数概率列表
            finish_reason=finish_reason,  # 生成结束的原因列表
        )