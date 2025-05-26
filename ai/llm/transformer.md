# Transformer

## 概述
其实可以把transformer理解成如下几个概要:
- vob字典里每个token都有一个特征向量(如768/1024), 基本代表了这个单词如 love 的特征
- pos位置里则是记录某个位置常见的特征向量(长度同上, 如768/1024), 表示这个位置如 第一个位置一般为 主语
- 而 attention自注意力 则表示 tok_emb+pos_emb 生成特征向量后的关系(q k v shape都是768*768), 也就是可以理解为: sequence序列里 每个单词之间的相互关系学习

供学习的代码可参考: [myGPT实现](https://github.com/leafan/myGPT)

## Transformer 基础实现

![transformer网络图](https://i-blog.csdnimg.cn/blog_migrate/394142aa34b101d092b29c351f480335.png)

### Input Embedding + Positional Encoding
对应图中左下角的 Inputs与Positional Encoding 转换

1. 首先将输入token串转换成token位置嵌入矩阵
2. 然后 升维至 [token length(1024)][embeded layer], 将位置编码对应的特征信息[embeded]引入


### 自注意力(Multi-Head Attention)实现
对应图中的 Multi-Head Attention部分.

具体逻辑请查看代码与注释， 基本就是论文中关于自注意力的一个代码实现

```python

class CausalSelfAttention(nn.Module):
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # ​PyTorch的nn.Linear规则​：nn.Linear(in_features, out_features)
        # 的权重矩阵形状为(out_features,in_features)
        # 因此，当定义nn.Linear(C, 3C)时, 权重矩阵W的形状为(3C,C)

        # 把输入经过线性层得到的矩阵切分，得到qkv三个矩阵, c_attn的定义:
        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 相当于 c_attn 便是 qkv本身的权重矩阵, 在这里运算后, 将x也一起相乘计算好了
        # W是形状为(3C,C)的权重矩阵, WT是其转置, 形状(C,3C)
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # 再将k，q，v的embedding分给每个head
        # k和q负责计算attention score，v是每个token的embedding
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 计算了每一对token的Q和K的缩放点积，从而得到每对token之间的attention score
        # att矩阵形状： (batch_size, n_head, sequence_length, sequence_length) 
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # mask操作，使得每个token和它自己之后token计算出来的attention score被mask掉
        # 从而不会让前面的token得到后面token的相关信息
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # 归一化得到0-1之间的score
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 通过attention score乘上每个token的embedding，每个token的embedding为它
        # 所在的sequence embedding的加权和
        # 这样每个embedding都得到了关于整个句子的信息
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)


        # 最后把n个head连起来，恢复维度
        # (batch_size, sequence_length, embedding_dimensionality)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
```


### Add&Norm 和 Feed Forward
这两个组合起来本质就类似于一个残差网络块(ResNet, Residual Network)

#### Feed Forward
名字叫前馈网络FFN(Feed Forward Network), 逻辑是先升维再降维, 增加网络参数容量; 引入非线性,与自注意力层互补, 增强模型表达能力.

在代码实现中叫 MLP(Multi-Layer Perceptron), 含义是一样的:

```python

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)  # 升维
        x = self.gelu(x)
        x = self.c_proj(x) # 降维
        x = self.dropout(x) # 防过拟合
        return x
```

#### Add&Norm
这是残差的一个实现, 比较简单, 也是图中箭头的含义, 实现:

```python
class Block(nn.Module):

    def forward(self, x):
        # 通过与自身相加, 实现残差
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

```


## 参考
1. https://zhuanlan.zhihu.com/p/601044938 —— 对nanoGPT的源码分析