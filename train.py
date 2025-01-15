import os
from tqdm.autonotebook import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from nets.transformer import poemTransformer
from utils.dataloader import PoetryDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###### 训练transformer模型 ######

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train Transformer model from scratch.")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file')
    args = parser.parse_args()

    checkpoint_path = args.checkpoint    
    os.makedirs("models", exist_ok=True)  # 创建models文件夹
    # d_model = 512   # 字 Embedding 的维度
    # d_ff = 2048     # 前向传播隐藏层维度
    # d_k = d_v = 64  # K(=Q), V的维度 
    # n_layers = 6    # 有多少个encoder和decoder
    # n_heads = 8     # Multi-Head Attention设置为8
    
    dataset = PoetryDataset(split="train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    vocab_size = dataset.vocab_size 
    model = poemTransformer(vocab_size)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)     #忽略 占位符 索引为0.
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    model.to(device)
    epoch_num = 10
    for epoch in range(epoch_num):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(loader), desc=f'Epoch {epoch + 1}/{epoch_num}', unit='batch') as pbar:
            for enc_inputs, dec_inputs, dec_outputs in loader:
                # enc_inputs : [batch_size, src_len]  # dec_inputs : [batch_size, tgt_len] # dec_outputs: [batch_size, tgt_len] 
                enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
                outputs = model(enc_inputs, dec_inputs) # outputs: [batch_size * tgt_len, tgt_vocab_size]
                loss = criterion(outputs.view(-1, outputs.size(-1)), dec_outputs.view(-1))
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                pbar.update(1)
        print(f'Epoch {epoch + 1} loss = {epoch_loss / len(loader):.6f}')

        # 保存模型参数,几个epoch就接近收敛了
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), os.path.join('./models', f'model_epoch_{epoch + 1}.pth'))