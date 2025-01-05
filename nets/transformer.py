import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Callable, Optional, Union
from torch import Tensor

###### 定义Transformer主体架构 ######

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #->(max_len, model_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # inpur shape: [batch_size, seq_len, d_model]，batch_first implement
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

# 定义自注意力机制
class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, attn_drop = 0., proj_drop = 0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads  # 多头注意力机制的头数
        self.head_dim = dim // num_heads  # 每个头的维度
        self.scale = self.head_dim ** -0.5  # 归一化参数

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)  # 产生qkv
        self.W_Q = nn.Linear(dim, self.head_dim * num_heads, bias=qkv_bias)
        self.W_K = nn.Linear(dim, self.head_dim * num_heads, bias=qkv_bias)
        self.W_V = nn.Linear(dim, self.head_dim * num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # attention_score的dropout
        self.proj = nn.Linear(dim, dim)  # 多头注意力合并之后的语义空间转化
        self.proj_drop = nn.Dropout(proj_drop)  # 输出的dropout
    
    def forward(self, q, k, v, mask = None):
        B, N, _ = q.shape  # bach_size的大小，sequence的长度， 每个token的维度

        Q = self.W_Q(q).view(B, -1, self.num_heads, self.head_dim).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(k).view(B, -1, self.num_heads, self.head_dim).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(v).view(B, -1, self.num_heads, self.head_dim).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
    
        attn = (Q @ K.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn.masked_fill_(mask == 0, -1e9)  # 将mask为0的位置设置为负无穷
        
        attn = attn.softmax(dim = -1)  # 获取归一化后的attention_score
        attn = self.attn_drop(attn)
        
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        x = (attn @ V).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 定义MLP结构
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features , activation, dropout = 0.):
        super(MLP, self).__init__()
        out_features = in_features

        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一层全连接层
        self.act = activation  # 激活函数
        self.drop1 = nn.Dropout(dropout)  # 随机dropout
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二层全连接层
        self.drop2 = nn.Dropout(dropout)

        self._init_weights()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def _init_weights(self):
        """
        Initialize the weights of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用正态分布初始化线性层的权重
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    # 将偏置初始化为0
                    nn.init.constant_(m.bias, 0)

class EncoderLayer(nn.Module):
    def __init__(self, model_dim = 512, num_heads = 8, ffn_dim = 2048, activation=F.relu, qkv_bias = False, dropout = 0., attn_drop = 0.):
        super(EncoderLayer, self).__init__()
        self.attn = Attention(model_dim, num_heads)
        self.ffn = MLP(model_dim, ffn_dim, activation, dropout)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.dropout1(self.attn(x, x, x, mask))
        x = self.ln1(x)
        x = x + self.dropout2(self.ffn(x))
        x = self.ln2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, model_dim = 512, num_heads = 8, ffn_dim = 2048, activation=F.relu, qkv_bias = False, dropout = 0., attn_drop = 0.):
        super(DecoderLayer, self).__init__()
        self.attn1 = Attention(model_dim, num_heads)
        self.attn2 = Attention(model_dim, num_heads)
        self.ffn = MLP(model_dim, ffn_dim, activation, dropout)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.ln3 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask, enc_mask):
        x = x + self.dropout1(self.attn1(x, x, x, self_mask))
        x = self.ln1(x)
        x = x + self.dropout2(self.attn2(x, enc_out, enc_out, enc_mask))
        x = self.ln2(x)
        x = x + self.dropout3(self.ffn(x))
        x = self.ln3(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, encoder_depth, model_dim, num_heads, ffn_dim, activation, qkv_bias, dropout, attn_drop):
        super(EncoderBlock, self).__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(model_dim, num_heads, ffn_dim, activation, qkv_bias, dropout, attn_drop) for _ in range(encoder_depth)
        ])
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, decoder_depth, model_dim, num_heads, ffn_dim, activation, qkv_bias, dropout, attn_drop):
        super(DecoderBlock, self).__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(model_dim, num_heads, ffn_dim, activation, qkv_bias, dropout, attn_drop) for _ in range(decoder_depth)
        ])
    
    def forward(self, x, enc_out, self_mask, enc_mask):
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, enc_mask)
        return x

class TransformerModel(nn.Module):
    def __init__(self,src_vocab_size, tgt_vocab_size, src_pad_idx=0, trg_pad_idx=0, model_dim=512, encoder_depth=3, decoder_depth=3, num_heads=8, ffn_dim=2048, qkv_bias=False, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, dropout=0., attn_drop_rate=0.):
        super(TransformerModel, self).__init__()
        
        self.src_emb = nn.Embedding(src_vocab_size, model_dim) 
        self.tgt_emb = nn.Embedding(tgt_vocab_size, model_dim)
        self.pos_emb = PositionalEncoding(model_dim)

        # 定义多个block
        self.encoderblock = EncoderBlock(encoder_depth, model_dim, num_heads, ffn_dim, activation, qkv_bias, dropout, attn_drop_rate)

        # 定义多个block
        self.decoderblock = DecoderBlock(decoder_depth, model_dim, num_heads, ffn_dim, activation, qkv_bias, dropout, attn_drop_rate)
        
        self.norm = nn.LayerNorm(model_dim)
        self.out = nn.Linear(model_dim, tgt_vocab_size)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.init_weights()

    def forward(self, src, tgt):
        # src_mask: [1, src_len, src_len]
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_trg_mask(tgt)

        src = self.src_emb(src)
        tgt = self.tgt_emb(tgt)
        src = self.pos_emb(src)
        tgt = self.pos_emb(tgt)

        enc_out = self.encoderblock(src, src_mask)
        dec_out = self.decoderblock(tgt, enc_out, tgt_mask, src_mask)
        out = self.out(dec_out)
        return F.log_softmax(out, dim=-1)
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用正态分布初始化线性层的权重
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    # 将偏置初始化为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                # 将 LayerNorm 的权重初始化为1，偏置初始化为0
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

def minitransformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(src_vocab_size=10, tgt_vocab_size=10, src_pad_idx=0, trg_pad_idx=0, model_dim=512, encoder_depth=3, decoder_depth=3, num_heads=8, ffn_dim=2048, qkv_bias=False, device=device, dropout=0., attn_drop_rate=0.)
    return model

def poemTransformer(vocab_size):
    model = TransformerModel(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, model_dim=512, encoder_depth=3, decoder_depth=3, num_heads=8, ffn_dim=2048, qkv_bias=False)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9], [1, 8, 7, 3, 4, 5]]).to(device)
    tgt = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = TransformerModel(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    out = model(x, tgt[:, :-1])
    print(out.shape)