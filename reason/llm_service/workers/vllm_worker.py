"""
A model worker that executes the model based on vLLM.

See documentations at docs/vllm_integration.md
"""

import argparse
import asyncio
import json
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from reason.llm_service.workers.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import get_context_length


app = FastAPI()


class VLLMWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,           # 控制器地址
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,  # 限制工作节点的并发数
        no_register: bool,              # 是否不进行注册
        llm_engine: AsyncLLMEngine,     # 管理模型加载、推理
        conv_template: str,             # 用于对话模型的处理或生成策略
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: vLLM worker..."
        )
        self.tokenizer = llm_engine.engine.tokenizer.tokenizer
        self.context_len = get_context_length(llm_engine.engine.model_config.hf_config) # 上下文长度（模型能够处理的最大输入文本长度）

        if not no_register: # 工作节点失效或崩溃
            self.init_heart_beat()

    # 文本生成
    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        n = params.get("n", 1)                                      # 生成的样本数量，默认为 1
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))       # 控制生成文本的随机性（值越低越确定）
        top_p = float(params.get("top_p", 1.0))                   # 控制累积概率的阈值（值越低越确定）Set to 1 to consider all tokens
        top_k = params.get("top_k", -1.0)                           # 候选词的数量
        presence_penalty = float(params.get("presence_penalty", 0.0))       # 避免重复生成
        frequency_penalty = float(params.get("frequency_penalty", 0.0))     # 减少频繁出现的单词的概率
        max_new_tokens = params.get("max_new_tokens", 256)          # 生成的最大 token 数
        stop_str = params.get("stop", None)                         # 停止输出标志
        stop_token_ids = params.get("stop_token_ids", None) or []   # 特定的 token ID 列表，用于停止生成
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        echo = params.get("echo", True)                             # 返回的文本是否包括prompt
        use_beam_search = params.get("use_beam_search", False)
        best_of = params.get("best_of", None)
        include_stop_str_in_output = params.get("include_stop_str_in_output", False) # 是否将停止字符串包括在输出中

        # Handle stop_str
        stop = set() # 字符串或字符串列表转换为集合
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        # for tid in stop_token_ids:
        #     if tid is not None:
        #         stop.add(self.tokenizer.decode(tid))

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0 # Set to 1 to consider all tokens

        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            use_beam_search=use_beam_search,
            stop=list(stop),
            stop_token_ids=stop_token_ids,
            max_tokens=max_new_tokens,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of=best_of,
            logprobs=1,
            include_stop_str_in_output=include_stop_str_in_output,
        )
        results_generator = engine.generate(context, sampling_params, request_id) # 生成

        # 异步逐步返回生成结果
        async for request_output in results_generator:
            prompt = request_output.prompt
            # 根据 echo 参数，是否将 prompt 添加到生成的文本前
            if echo:
                text_outputs = [
                    prompt + output.text for output in request_output.outputs
                ]
            else:
                text_outputs = [output.text for output in request_output.outputs]
            # text_outputs = " ".join(text_outputs)
            # Note: usage is not supported yet
            # 计算token数量
            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = sum(
                len(output.token_ids) for output in request_output.outputs
            )
            # 返回结果
            ret = {
                "text": text_outputs,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "cumulative_logprob": [
                    output.cumulative_logprob for output in request_output.outputs
                ],
                "output_token_len": [
                    len(output.token_ids) for output in request_output.outputs
                ],
                "finish_reason": (
                    request_output.outputs[0].finish_reason
                    if len(request_output.outputs) == 1
                    else [output.finish_reason for output in request_output.outputs]
                ),
            }
            # 返回JSON 编码后的字符串
            # \0 通常用于流式传输，确保每个生成的文本片段是一个完整的、可解析的单位
            yield (json.dumps(ret) + "\0").encode()

    # 去掉末尾的流结束符 \0
    # 解析并返回完整的生成结果，有效地处理较大或长时间生成的内容
    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass # 不做处理
        return json.loads(x[:-1].decode())

# 释放信号量，表示一个任务完成，可以允许其他任务执行
def release_worker_semaphore():
    worker.semaphore.release()

# 管理并发控制，确保执行的任务数不会超过某个限制
def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()

# 创建并返回一个包含两个后台任务的 BackgroundTasks 对象，分别是释放信号量和中止请求
def create_background_tasks(request_id):
    async def abort_request() -> None:
        await engine.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks

# 流式数据生成任务的 API
@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)

# 非流式数据生成的 API
@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    output = await worker.generate(params)
    release_worker_semaphore() # 任务完成，释放信号量
    await engine.abort(request_id)
    return JSONResponse(output)

# 获取状态 API
@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()

# 计算 token 数量 API
@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)

# 获取对话模板 API
@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()

# 获取上下文长度 API
@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser() # 处理命令行输入
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    ) # 多个模型名称
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024) # 限制工作节点的并发处理数
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1) # GPU 数量
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) when"
        "downloading the model and tokenizer.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="The ratio (between 0 and 1) of GPU memory to"
        "reserve for the model weights, activations, and KV cache. Higher"
        "values will increase the KV cache size and thus improve the model's"
        "throughput. However, if the value is too high, it may cause out-of-"
        "memory (OOM) errors.",
    ) # 模型的 GPU 内存比例(过高会导致内存溢出（OOM）)

    parser = AsyncEngineArgs.add_cli_args(parser) # 额外的命令行参数(与引擎相关的配置)
    args = parser.parse_args() # 所有用户传入的命令行参数及其值
    if args.model_path:
        args.model = args.model_path
    if args.num_gpus > 1:
        args.tensor_parallel_size = args.num_gpus

    engine_args = AsyncEngineArgs.from_cli_args(args)
    # 根据转换后的引擎参数，初始化一个异步的 LLM（大语言模型）引擎
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    # 工作节点对象，负责与控制节点通信并执行计算任务
    worker = VLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        engine,
        args.conv_template,
    )
    # 启动 Uvicorn 服务器
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
