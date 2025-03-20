import torch

# tokenizer分词器分词 encode()编码

# 分析模型对特定步骤的预测效果
# 感觉就是算那个value，打分
# 从模型的输出中提取特定步骤（由 STEP_TAG 标记）对应的概率分布。
# 通过与 GOOD_TOKEN 和 BAD_TOKEN 配对的 logits，函数能够分析并返回模型对输入字符串中与 STEP_TAG 相关部分的推理结果。
@torch.inference_mode()  # 禁用梯度计算，提高推理速度
def _qwen_math_infer_fn(input_str: str, model, tokenizer, device):
    GOOD_TOKEN = '+'
    BAD_TOKEN = '-'
    STEP_TAG = '\n\n\n\n\n'  # 步骤标记

    # 编码，参与模型的计算的转化为张量，并放置到指定设备上
    # f" {GOOD_TOKEN} {BAD_TOKEN}"格式化字符串，中间有一个空格
    candidate_tokens = tokenizer.encode(f" {GOOD_TOKEN} {BAD_TOKEN}")  # [488, 481]
    step_tag_id = torch.tensor([tokenizer.encode(f" {STEP_TAG}")], device=device)  # 76325
    input_id = torch.tensor([tokenizer.encode(input_str)], device=device)
    # 模型推理，只关心与 GOOD_TOKEN 和 BAD_TOKEN 相关的未归一化的预测得分
    logits = model(input_id).logits[:,:,candidate_tokens]  # logits.shape: (batch_size样本, seq_length位置, num_classes类别数)

    # softmax归一化  softmax(dim=-1)得到 GOOD_TOKEN 和 BAD_TOKEN 的概率分布
    # scores[:,:,0] 提取了归一化后的得分，假设我们只关心第一类（例如 GOOD_TOKEN），得到每个 token 在该类别下的概率
    scores = logits.softmax(dim=-1)[:,:,0]  # 对最后一维（类别维度）进行 softmax 归一化，得到每个 token 的概率
    # 创建一个掩码，找出 input_id 中等于 step_tag_id 的位置
    mask = input_id == step_tag_id  # 通过比较每个 token 是否等于 STEP_TAG ID 来生成一个布尔 mask
    # 通过 mask 从 scores 中提取对应位置的得分，这里是从 softmax 结果中提取 STEP_TAG 的概率
    step_scores = scores[mask]  # 根据 mask 从 scores 中选择相应的得分，返回一个 tensor，其中包含了与 STEP_TAG 对应的概率分布
    return step_scores


@torch.inference_mode()  # 禁用梯度计算，提高推理速度
def _math_shepherd_infer_fn(input_str: str, model, tokenizer, device):
    GOOD_TOKEN = '+'
    BAD_TOKEN = '-'
    STEP_TAG = 'ки'   # 设定步骤标签（例如俄语字符）
    # 编码 GOOD_TOKEN 和 BAD_TOKEN，忽略第一个 token（通常是一个特殊的标记）
    candidate_tokens = tokenizer.encode(f"{GOOD_TOKEN} {BAD_TOKEN}")[1:]  # [648, 387]
    # 获取 STEP_TAG 对应的 token ID
    step_tag_id = tokenizer.encode(f"{STEP_TAG}")[-1]  # 12902

    # 将输入字符串转换为 token ID，放置到指定设备（CPU 或 GPU）
    input_id = torch.tensor([tokenizer.encode(input_str)], device=device)
    # 获取模型的输出 logits，选择与 GOOD_TOKEN 和 BAD_TOKEN 对应的 logits
    logits = model(input_id).logits[:,:,candidate_tokens]
    # 对 logits 进行 softmax，计算每个 token 的概率
    scores = logits.softmax(dim=-1)[:,:,0] 
    # 提取输入中与 STEP_TAG 相匹配位置的分数
    step_scores = scores[input_id == step_tag_id]
    # 返回与 STEP_TAG 匹配位置的分数
    return step_scores