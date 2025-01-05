import os
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# 获取项目根目录
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1] 

class PoetryDataset(Dataset):
    def __init__(self, split="train", max_length=64):
        """
        :param split: 数据集的分割，"train" 或 "test"
        :param max_length: 最大序列长度
        """
        super(PoetryDataset, self).__init__()
        
        # 使用get_project_root函数来获得项目根目录
        root_dir = get_project_root()
        
        # 硬编码的数据集路径，相对于项目根目录
        data_dir = {
            "train": root_dir / "datasets" / "train",
            "test": root_dir / "datasets" / "test"
        }
        
        if split not in data_dir:
            raise ValueError("split must be either 'train' or 'test'")
            
        # 加载和解析JSON文件
        self.data = []
        for filename in os.listdir(data_dir[split]):
            if filename.endswith('.json'):
                with open(os.path.join(data_dir[split], filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.data.extend(self._extract_poems(data))
        
        # 初始化分词器
        bert_model_dir = root_dir / "plugins" / "bert-base-chinese"
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir) 
        self.vocab_size = self.tokenizer.vocab_size
        self.max_length = max_length

        self.processed_data = self._process_data(self.data)
        
    def _extract_poems(self, data):
        """从数据中提取所有的诗句"""
        poems = []
        for item in data:
            paragraphs = item.get("paragraphs", [])
            for paragraph in paragraphs:
                if paragraph.strip():  # 确保不是空行
                    poems.append(paragraph)
        return poems

    def _process_data(self, poems):
        processed_data = []
        for poem in poems:
            sentences = poem.split('，')
            if len(sentences) < 2:
                continue            
            enc_input = sentences[0]
            dec_output_base = sentences[1]

            for i in range(1, len(dec_output_base)+1):
                dec_input = dec_output_base[:i-1]
                dec_output = dec_output_base[:i] + ("" if i < len(dec_output_base) else "[SEP]")
                
                enc_input_tokenized = self.tokenizer.encode(enc_input, max_length=self.max_length, padding='max_length', truncation=True)
                dec_input_tokenized = self.tokenizer.encode(dec_input, max_length=self.max_length, padding='max_length', truncation=True)
                dec_output_tokenized = self.tokenizer.encode(dec_output, max_length=self.max_length, padding='max_length', add_special_tokens=False, truncation=True)

                processed_data.append((enc_input_tokenized, dec_input_tokenized, dec_output_tokenized))
        
        return processed_data

    def __getitem__(self, index):
        """
        获取数据集中的单个样本
        按照数据处理尽量放在dataloader部分的原则,这里将数据处理成了模型输入的格式
        :param index: 索引
        :return: 编码器输入、解码器输入和目标输出
        """
        enc_inputs, dec_inputs, dec_outputs = self.processed_data[index]
        return torch.tensor(enc_inputs), torch.tensor(dec_inputs), torch.tensor(dec_outputs)

    def __len__(self):
        """
        获取数据集的大小
        :return: 数据集样本数量
        """
        return len(self.processed_data)

# 单元测试
if __name__ == "__main__":
    # 创建训练集和测试集
    train_dataset = PoetryDataset(split="train")
    test_dataset = PoetryDataset(split="test")
    # 打印一些信息以验证数据集是否正确构建
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")