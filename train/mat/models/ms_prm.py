from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from torch import nn
import numpy as np
from mat.envs.math.prompts import IN_CONTEXT_EXAMPLE

class MSProcessRM(nn.Module):

    def __init__(self, all_args):
        super().__init__()
        self.model_name_or_path = all_args.prm_model_name_or_path
        
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = 'ки' # 步骤标记

        # 使用预训练的tokenizer进行文本分词
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.pad_token_id = 0
        # 编码good_token和bad_token
        self.candidate_tokens = self.tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:] # [648, 387]
        self.step_tag_id = self.tokenizer.encode(f"{self.step_tag}")[-1] # 12902
        # 加载预训练的语言模型
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, device_map="auto",).eval()
        print(f"successfully loaded PRM.")
            
    # 计算给定状态（观察）和动作的奖励分数
    @torch.no_grad()
    def get_reward(self, obs: list[np.ndarray[str]], actions: list[np.ndarray[str]]):
        inputs_for_prm = []
        # 遍历每个状态和动作
        for o, a in zip(obs.copy(), actions.copy()):
            # 去除无关的上下文示例
            o = o[0].replace(IN_CONTEXT_EXAMPLE, "")
            # 去除步标签并去除多余空格
            a = a[0].replace(self.step_tag, "").strip()
            inputs_for_prm.append(f"{o}{a} {self.step_tag}")

        # 将输入编码为模型可处理的格式，并将其送入GPU
        input_ids = self.tokenizer(inputs_for_prm, return_tensors="pt", padding=True).to("cuda")
        # 获取模型的输出logits，提取与候选token相关的logits
        logits = self.model(**input_ids).logits[:, :, self.candidate_tokens]
        # 对logits进行softmax运算，计算出概率分布
        score = logits.softmax(dim=-1)[:, :, 0]
        
        step_scores = [] # 存储每个样本的奖励分数
        for i in range(np.shape(score)[0]):
            # 找到与step_tag_id对应的位置，并提取该位置的分数
            step_score = score[i][input_ids["input_ids"][i] == self.step_tag_id]
            # 提取最后一步的分数
            last_step_score = step_score[-1]
            step_scores.append([last_step_score.item()])
        step_scores = np.array(step_scores)
        
        return step_scores
