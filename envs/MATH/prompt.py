COT_EXAMPLES = None
COT_TASK_DESC = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>"
PROBLEM_FORMAT_STR = """<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"""

SEP = "\n\n"

# 这段代码的目的是为一个基于推理链（Chain of Thought, COT）任务的处理框架提供格式化工具。具体来说：

# 它会提示用户逐步推理，并在最终结果中使用 \boxed{} 格式。
# 它定义了如何格式化用户输入的问题，以及模型回答的结构。
# 使用分隔符来区分不同的段落或数据块。