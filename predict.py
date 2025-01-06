from pathlib import Path
import argparse
import torch
from transformers import BertTokenizer
from nets.transformer import poemTransformer
# from utils.dataloader import PoetryDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path, vocab_size):
    model = poemTransformer(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def predict(model, tokenizer, enc_input, max_length):
    enc_input_tokenized = tokenizer.encode(enc_input, max_length=max_length, truncation=True)
    enc_input_tensor = torch.tensor(enc_input_tokenized).unsqueeze(0).to(device)  # 添加batch维度

    result = ""
    dec_input_tokenized = tokenizer.encode(result, max_length=max_length, truncation=True)
    dec_input_tensor = torch.tensor(dec_input_tokenized).unsqueeze(0).to(device)  # 添加batch维度


    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(enc_input_tensor, dec_input_tensor)
            next_token = outputs.argmax(dim=-1)[:, -2].item()# top1取词
            if next_token == tokenizer.sep_token_id:
                break
            nestchar = tokenizer.decode([next_token], skip_special_tokens=True)
            if nestchar == '。':
                break
            result += nestchar
            dec_input_tokenized = tokenizer.encode(result, max_length=max_length, truncation=True)
            dec_input_tensor = torch.tensor(dec_input_tokenized).unsqueeze(0).to(device)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the next poem sentence using the trained Transformer model.")
    parser.add_argument('--model', type=str, default='./models/model_epoch_10.pth', help='Path to the model file')
    args = parser.parse_args()

    model_path = args.model
    max_length = 50   

    # 初始化分词器
    root_dir = Path(__file__).resolve().parents[0] #项目根目录
    bert_model_dir = root_dir / "plugins" / "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
    vocab_size = tokenizer.vocab_size

    # 加载模型
    model = load_model(model_path, vocab_size)

    # 输入句子
    while True:
        enc_input = input("请输入诗的前半句: ")
        result = predict(model, tokenizer, enc_input, max_length)
        print(f"输入: {enc_input}")
        print(f"输出: {result}")