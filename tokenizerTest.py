from pathlib import Path
from transformers import BertTokenizer

root_dir = Path(__file__).resolve().parents[0] #项目根目录
bert_model_dir = root_dir / "plugins" / "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
max_length = 50

while True:
    enc_input = input("请输入代编码句子: ")
    result = tokenizer.encode(enc_input, max_length=max_length, truncation=True)
    print(f"输出: {result}")

# while True:
#     dec_input = input("请输入代解码列表（用逗号分隔的整数）: ")
#     try:
#         # 将输入的字符串转换为整数列表
#         dec_input_list = [int(x) for x in dec_input.split(',')]
#         result = tokenizer.decode(dec_input_list, max_length=max_length, truncation=True)
#         print(f"输出: {result}")
#     except ValueError:
#         print("输入格式错误，请输入用逗号分隔的整数列表。")