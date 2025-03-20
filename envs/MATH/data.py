from pathlib import Path
import jsonlines
from torch.utils.data import Dataset


def get_train_test_dataset(*args, **kwargs):
    env_dir = Path(__file__).parent # 获取当前文件所在目录
    test_ds = JsonlMathDataset(env_dir / "dataset/test500.jsonl") # 创建测试数据集（从 test500.jsonl 文件加载）
    train_ds = JsonlMathDataset(env_dir / "dataset/train.jsonl")  # 创建训练数据集（从 train.jsonl 文件加载）
    return train_ds, test_ds


class JsonlMathDataset(Dataset):
    def __init__(self, data_path):
        super().__init__() # 调用父类构造函数初始化
        self.data = [] # 初始化数据列表

        # 从给定的 JSONL 文件中读取数据
        with jsonlines.open(data_path, "r") as reader:
            for obj in reader:
                self.data.append(obj)

    def __len__(self):
        # 返回数据集的大小（样本数量）
        return len(self.data)

    def __getitem__(self, index):
        # 获取指定索引的数据项，返回问题和答案
        x = self.data[index]
        return {"question": x["problem"], "answer": x["solution"]}
