from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import argparse
import os

from peft import PeftModel
from peft import get_peft_model, LoraConfig, TaskType
# Ensure bitsandbytes is available for 8-bit quantization
# import bitsandbytes as bnb
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

from torch.nn import BCEWithLogitsLoss
from transformers import DataCollatorWithPadding
from datasets import concatenate_datasets

import random

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/mnt/data101_d2/wangzhu/llm_models/Qwen2.5-Math-7B-Instruct")
parser.add_argument("--data_path", type=str, default="/mnt/data101_d2/wangzhu/datasets")
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
parser.add_argument("--total_batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--datasets", type=str, default='both')
parser.add_argument("--server", type=str, default='1')


args = parser.parse_args()


good_token = '+'
bad_token = '-'
step_tag = '\n\n\n\n\n' #ки
step_tag2 = '\n\n'

model_path = args.model_path

# tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    add_eos_token=False, 
)

print(tokenizer.encode('a ки b'))
print(tokenizer.encode('a b'))

print(tokenizer.encode('a \n\n b'))
print(tokenizer.encode('a b'))
print(tokenizer.encode('a \n\n\n\n\n\n b'))
print(tokenizer.encode('a b'))


print(tokenizer.encode('a \n\n\n\n\n\n\n b'))
print(tokenizer.encode('a b'))

print(tokenizer.encode('a \n\n\n\n\n\n\n\n b'))
print(tokenizer.encode('a b'))


print(tokenizer.encode('a + b'))
print(tokenizer.encode('a b'))

print(tokenizer.encode('a - b'))
print(tokenizer.encode('a b'))

print(tokenizer.encode(' + -'))
print(tokenizer.encode('+-'))


# if USE_8bit is True:
#     model = prepare_model_for_int8_training(model)
# 打印分词器的结束标记ID（eos_token_id）
print(tokenizer.eos_token_id)

# 设置填充标记ID为0，表示未知词汇（unk），与结束标记ID不同
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
# 设置填充方式为“左侧填充”（即从序列的左边开始填充）
tokenizer.padding_side = "left"  # Allow batched inference


# tokenizer = AutoTokenizer.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm')
# 使用tokenizer对给定的tokens进行编码，返回token的ID列表
candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}") # [488, 481]
print(candidate_tokens)
# 获取某个step标记的ID，编码后的最后一个ID（通常是一个特定的步骤标签）
step_tag_id = tokenizer.encode(f" {step_tag}")[-1] # 76325 # 获取编码后最后一个ID
print('step_tag_id:',tokenizer.encode(f" {step_tag}"))
print('step_tag_id2:',tokenizer.encode(f"{step_tag2}"))
# model = AutoModelForCausalLM.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm').eval()
# model = AutoModelForCausalLM.from_pretrained(model_path).eval()
# 加载一个训练好的语言模型，这里选择的是`AutoModelForCausalLM`，并设置一些额外的参数（如数据类型和设备映射）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # load_in_8bit=True,   # Enables 8-bit quantization
    # device_map="auto",   # Automatically assigns the model to available GPUs/CPUs
    # torch_dtype=torch.float16,  # Mixed precision for faster inference
    torch_dtype=torch.bfloat16,  # 设置使用bfloat16进行推理（适合FP16的硬件）
    attn_implementation="flash_attention_2",   # 使用FlashAttention加速
)

# for name,param in model.named_parameters():
#     print(name)
# 输出加载的模型信息
print(model)

# LoRA配置，用于适应特定的语言任务（如因果语言建模）
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # LoRA for causal language modeling task# 任务类型为因果语言建模
    r=8,  # Rank of LoRA  # LoRA的秩（控制降维的大小）
    lora_alpha=32,  # Alpha scaling factor for LoRA # LoRA的alpha系数，用于缩放
    lora_dropout=0.1,  # Dropout rate for LoRA layers # LoRA层的dropout率
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific layers # 目标模块，LoRA只应用于这些模块
)

# 使用LoRA微调模型
model = get_peft_model(model, lora_config)

# model.to('cuda:0') # 可以将模型移动到GPU上
# 输出模型当前使用的设备
print(model.device)
# 设定一个问题，模型将基于问题生成推理步骤
question = "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
# 设定两个输出，分别是正确的和错误的推理步骤（答案）
output1 = "Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки" # 18 is right
output2 = "Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки" # 17 is wrong
def preprocess_function(example):
    # 构造输入字符串，包含问题和处理步骤
    input = f"{example['question']} {example['process']}"
    # 对输入进行分词处理，设置最大长度为2048
    tokenized_inputs = tokenizer(
        input, 
        truncation=True, 
        padding='max_length', 
        # padding=True,
        max_length=2048,
    )
    
    def find_all_indices(lst, element):
        return [i for i, x in enumerate(lst) if x == element]
    
    length = len(tokenized_inputs['input_ids'])
    # print(length)
    # 查找所有step_tag_id在input_ids中的位置
    indices = find_all_indices(tokenized_inputs['input_ids'],step_tag_id)
    
    # 调整标签长度，确保标签数量与step_tag_id的位置数量一致
    if len(indices) != len(example['label']):
        # print(example)
        example['label'] = example['label'][:len(indices)]
    
    assert len(indices) == len(example['label'])
    
    # 初始化标签为-100，标记需要忽略的token
    tokenized_inputs['labels'] = [-100] * length
    # tokenized_inputs['attention_mask'] = [1] *length
    # print(len(indices))
    # 根据标签修改token的label，并更新attention_mask
    for i in range(len(indices)):
        if example['label'][i] == '+' or example['label'][i] == 1:
            tokenized_inputs['labels'][indices[i]] = candidate_tokens[0]
        elif example['label'][i] == '-' or example['label'][i] == 0:
            tokenized_inputs['labels'][indices[i]] = candidate_tokens[1]
        else:
            raise ValueError('label is wrong')
        tokenized_inputs['attention_mask'][indices[i]] = 0  # 将step_tag_id的attention_mask设为0
    # tokenized_inputs['labels'] = [-100] *(length-1) + tokenized_inputs['input_ids'][length-1:]
    
    return tokenized_inputs

# 定义数据路径，根据不同的配置加载不同的数据文件
DATA_PATH = {
    # "train": 'multi-step.json', 
    # 'train': 'test.json',
    # "test": os.path.join(args.data_path, 'prm800k_test.json'),
    "test": os.path.join(args.data_path, 'phase2_test_processed.json'), # 测试集路径
    "train": os.path.join(args.data_path, "MATH_APS_clean.json"), # 训练集路径
    # "train": "../../datasets/processed_data/prm800k/data/phase2_train_new.jsonl",
    # "test": "../../datasets/prm800k-main/prm800k/data/phase2_test_new.jsonl",
    
}

# 加载数据集
dataset = load_dataset('json', data_files=DATA_PATH)
# 如果选择加载“both”数据集，合并多个训练集
if args.datasets == 'both':
    dataset2 = load_dataset('json',data_files=os.path.join(args.data_path, "phase2_train_processed.json"))
    dataset['train'] = concatenate_datasets([dataset['train'], dataset2['train']])
# 如果选择加载“all”数据集，从多个数据源中加载数据并随机抽取样本
elif args.datasets == 'all':
    dataset2 = load_dataset('json',data_files=os.path.join(args.data_path, "phase2_train_processed.json"))
    dataset3 = load_dataset('json',data_files=os.path.join(args.data_path, "math-shepherd_cleaned.json"))

    aps_length = len(dataset['train'])
    prm800k_length = len(dataset2['train'])
    # 固定随机种子，随机选取各数据集的子集
    random.seed(42)
    dataset['train'] = dataset['train'].select(random.sample(range(aps_length),50000))
    random.seed(42)
    dataset2['train'] = dataset2['train'].select(random.sample(range(prm800k_length),50000))
    # 合并数据集
    dataset['train'] = concatenate_datasets([dataset['train'], dataset2['train'],dataset3['train']])
# 如果选择加载“aps_shepherd”数据集，合并特定的训练数据
elif args.datasets == 'aps_shepherd':
    dataset3 = load_dataset('json',data_files=os.path.join(args.data_path, "math_shepherd.json"))
    dataset['train'] = concatenate_datasets([dataset['train'],dataset3['train']])



# dataset['train'] = dataset['train'].select(range(200000,201000))
# dataset['test'] = dataset['test'].select(range(1000))

# 处理数据集，应用预处理函数
print('start processing')
tokenized_datasets = dataset.map(preprocess_function)
# 删除不需要的列（问题、过程、标签）
tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(['question','process','label'])
tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(['question','process','label'])

# 打印处理后的训练集信息
print(tokenized_datasets['train'])
print('dataset processed')
# print(tokenized_datasets['train']['input_ids'])
# print(len(tokenized_datasets['train']['input_ids'][0]))

# Data collator for padding inputs dynamically
# 初始化数据整理器，用于动态填充输入
data_collator = DataCollatorWithPadding(tokenizer)

# 设置批处理大小和梯度累积步数
BATCH_SIZE = args.total_batch_size
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // args.per_device_train_batch_size

# 判断是否使用分布式训练
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

# 打印训练环境信息
print(world_size)
print(ddp)

# 设置输出路径，包含批处理大小和学习率等配置
fp = f'bs_{args.total_batch_size}_lr_{args.learning_rate}_datasets_{args.datasets}'
output_path = f'./prm_results_qwen_new.{args.server}/{fp}'


# Training arguments
# 设置训练参数
training_args = TrainingArguments(
    output_dir=output_path,                                             # 设置输出路径
    evaluation_strategy="no",  # Evaluate at the end of each epoch      # 不进行每个epoch结束时的评估
    learning_rate=args.learning_rate,                                   # 学习率
    per_device_train_batch_size=args.per_device_train_batch_size,       # 每个设备的训练批处理大小
    per_device_eval_batch_size=args.per_device_eval_batch_size,         # 每个设备的评估批处理大小
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,            # 梯度累积步数
    num_train_epochs=3,                                                 # 训练的epoch数
    weight_decay=0.01,                                                  # 权重衰减（L2正则化）
    logging_dir="./logs",                                               # 日志存储目录
    logging_steps=10,                                                   # 每10步记录一次日志
    save_strategy="epoch",                                              # 按epoch保存模型
    # fp16=True,  # Enable mixed precision for better performance on supported hardware
    bf16=True,                                                          # 启用bfloat16精度训练
    report_to="none",                                                   # 不报告任何日志（可设置为"wandb"以使用Weights and Biases）
    # Set to "wandb" if you are using Weights and Biases for logging
    dataloader_num_workers=4,                                           # 数据加载时使用的子进程数量
    deepspeed=None,                                                     # 不使用DeepSpeed加速（可以传递配置以启用DeepSpeed）
    ddp_find_unused_parameters=False,                                   # 在分布式训练时，不自动查找未使用的参数
)

# Define a custom metric function (e.g., accuracy for binary classification)
# 定义自定义评估指标（例如：二分类任务的AUC、对数损失和准确率）
def compute_metrics(eval_pred):
    # pass
    # print(eval_pred)
    print('bb')
    # 获取预测结果和标签
    pre, labels = eval_pred
    # 计算AUC (Area Under the Curve)
    auc = roc_auc_score(pre[1], pre[0])
    # 计算对数损失
    ll = log_loss(pre[1], pre[0])
    # 计算准确率
    acc = accuracy_score(pre[1], pre[0] > 0.5)
    # 返回包含各个指标的字典
    result ={
        'auc': auc,  # AUC
        'll': ll,  # 对数损失
        'acc': acc,  # 准确率
    } 
    print(result)  # 打印结果
    return result

# 定义处理logits并计算自定义指标的函数
def preprocess_logits_for_metrics(logits,labels):
    # 打印调试信息
    print('aa')
    # return logits,labels
    # 获取标签为candidate_tokens[0]或candidate_tokens[1]的位置
    labels_index = torch.argwhere(torch.bitwise_or(labels == candidate_tokens[0], labels == candidate_tokens[1]))
    # 根据标签是否为candidate_tokens[1]，生成gold标签
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == candidate_tokens[1], 0, 1)
    # labels_index[: , 1] = labels_index[: , 1] - 1
    # 从logits中选取对应标签的位置
    logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [candidate_tokens[1], candidate_tokens[0]]]
    # 计算softmax概率
    prob = torch.softmax(logits, dim=-1)
    # 返回类别1的概率作为预测值，gold标签作为真实标签
    return prob[:, 1], gold
    

# Initialize the Trainer
trainer = Trainer(
    model=model,                                                 # 训练的模型
    args=training_args,                                          # 训练参数
    train_dataset=tokenized_datasets['train'],                   # 训练集
    eval_dataset=tokenized_datasets['test'],                     # 测试集（如果有验证集，替换为验证集）
    # Replace with a validation set if available
    data_collator=data_collator,                                 # 数据整理器
    tokenizer=tokenizer,                                         # 分词器
    preprocess_logits_for_metrics=preprocess_logits_for_metrics, # 自定义后处理函数
    compute_metrics=compute_metrics,                             # 自定义评估函数
)

# 开始训练
trainer.train()
# trainer.evaluate()

# Save the fine-tuned model and tokenizer
# 保存微调后的模型和分词器
model.save_pretrained('./fine_tuned_qwen_lora_8bit')
tokenizer.save_pretrained('./fine_tuned_qwen_lora_8bit')


# 遍历输出并计算分数
for output in [output1,output2]:  # 可以扩展为更多输出
# for output in [output1, output2,output3]:
    input_for_prm = f"{question} {output}"  # 合并问题和输出作为输入
    input_id = torch.tensor([tokenizer.encode(input_for_prm)])   # 转换为输入ID
    # print(input_id)

    with torch.no_grad():   # 禁用梯度计算
        logits = model(input_id).logits[:,:,candidate_tokens]  # 获取logits，选择候选标签
        # print(logits)
        scores = logits.softmax(dim=-1)[:,:,0]  # 对logits进行softmax，得到概率
        # print(scores)
        step_scores = scores[input_id == step_tag_id]  # 获取特定标签的概率
        
        print(step_scores)   # 打印特定标签的分数
        print('aaaaaa')   # 调试信息打印
# tensor([0.9955, 0.9958, 0.9983, 0.9957])
# tensor([0.9955, 0.9958, 0.9983, 0.0240])
