from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from torch import nn
from peft import PeftModel,PeftConfig
import numpy as np
from mat.envs.math.prompts import IN_CONTEXT_EXAMPLE

class QwenProcessRM(nn.Module):

    def __init__(self, all_args):
        super().__init__()
        self.model_name_or_path = all_args.prm_model_name_or_path
        self.prm_checkpoint_path = all_args.prm_checkpoint_path
        print(f"prm_base_model_path: {self.model_name_or_path}")
        print(f"prm_checkpoint_path: {self.prm_checkpoint_path}")
        
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = '\n\n\n\n\n' # 步骤划分

        # 使用预训练的分词器进行初始化，不自动添加结束符，并且填充方向为左
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, add_eos_token=False, padding_side='left')
         # 设置填充符的 token id
        self.tokenizer.pad_token_id = 151655 # "<|image_pad|>"
        # 编码奖励标记符号，用于后续的计算
        self.candidate_tokens = self.tokenizer.encode(f" {self.good_token} {self.bad_token}") # [488, 481]
        # 获取步骤标记符号的 id，用于标识每个步骤
        self.step_tag_id = self.tokenizer.encode(f" {self.step_tag}")[-1] # 76325
        # 加载预训练的语言模型（自动分配到GPU或CPU）
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, 
                                                          device_map="auto", 
                                                          torch_dtype=torch.bfloat16,
                                                        #   attn_implementation="flash_attention_2",
                                                          ).eval() # 设置为推理模式，不进行梯度计算
        
        # 加载微调后的模型（奖励模型），用于更精确的任务处理
        # PeftModel 是用于加载特定任务微调模型的类
        # adapter_config = PeftConfig.from_pretrained(cp_path)
        self.model = PeftModel.from_pretrained(self.model, self.prm_checkpoint_path)
        
    @torch.no_grad()
    def get_reward(self, obs: list[np.ndarray[str]], actions: list[np.ndarray[str]]):
        inputs_for_prm = [] # 存储模型输入文本

        # 遍历观测和动作列表，构造模型的输入文本
        for o, a in zip(obs.copy(), actions.copy()):
            # 移除上下文示例中的某些信息
            o = o[0].replace(IN_CONTEXT_EXAMPLE, "")
            # 替换原文本中的“ки”符号为步骤分隔符（step_tag）
            o = o.replace("ки", self.step_tag + " ")
            # 清理动作文本，去除“ки”并去掉多余的空格
            a = a[0].replace("ки", "").strip()
            # 将处理后的观测和动作文本组合成模型的输入
            inputs_for_prm.append(f"{o}{a} {self.step_tag}")

         # 使用分词器将文本转为 token ids，并将其移到 GPU 上进行处理
        input_ids = self.tokenizer(inputs_for_prm, return_tensors="pt", padding=True).to("cuda")

        # 通过模型进行推理，得到 logits 输出，选择与 `good_token` 和 `bad_token` 对应的部分
        logits = self.model(**input_ids).logits[:, :, self.candidate_tokens]
        # 使用 softmax 函数对 logits 进行归一化，得到奖励的概率分布
        score = logits.softmax(dim=-1)[:, :, 0] # 获取 `good_token` 的概率（`good_token` 是索引 0）(???)

        step_scores = []

        # 遍历每个样本，提取模型输出中的最后一个步骤的分数
        for i in range(np.shape(score)[0]):
            # 找到当前样本中与步骤标记符号 (`step_tag_id`) 对应的部分
            step_score = score[i][input_ids["input_ids"][i] == self.step_tag_id]
            # 获取该步骤的最后一个分数，作为该样本的奖励
            last_step_score = step_score[-1]
            # 将该分数添加到 step_scores 列表中
            step_scores.append([last_step_score.item()])

        # 将所有样本的奖励分数转换为 NumPy 数组
        step_scores = np.array(step_scores)
        
        return step_scores
