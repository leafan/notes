# YOLO12

YOLO12 引入了一种以注意力为中心的架构, 它不同于以往 YOLO 模型中使用的基于 CNN 的传统方法, 但仍保持了许多应用所必需的实时推理速度. 该模型通过对注意力机制和整体网络架构进行新颖的方法创新, 实现了最先进的物体检测精度, 同时保持了实时性能.

## YOLO12 改进点

- **区域注意机制：** 一种新的自我注意方法, 能有效处理大的感受野. 它可将特征图横向或纵向划分为 x 个大小相等的区域(默认为 4 个), 从而避免复杂的操作, 并保持较大的有效感受野. 与标准自注意相比, 这大大降低了计算成本

- **优化注意力架构:** 简化了标准关注机制, 以提高效率并与YOLO框架兼容. 这包括:
    + 使用 FlashAttention 尽量减少内存访问开销
    + 去除位置编码, 使模型更简洁、更快速
    + 调整 MLP 比例(从通常的 4 调整为 1.2 或 2), 以更好地平衡注意力层和前馈层之间的计算
    + 减少堆叠区块的深度, 提高优化效果
    + 酌情利用卷积运算, 提高计算效率
    + 在注意力机制中加入 7x7 可分离卷积(位置感知器), 对位置信息进行隐式编码

- **...**


## transformer实现

yolov12中使用 A2C2f 模块来实现注意力, 且在backbone与head中均多次使用.

先查看yolo配置文件:

```ini
# YOLO12-turbo backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,  [64, 3, 2]] # 0-P1/2
# - ...

  - [-1, 1, Conv,  [512, 3, 2]] # 5-P4/16
  
  # P4/16: 即输入图像下采样16倍后的特征图
  # 512: 输出通道数; True: 表示启用残差连接; 4: 区域注意力划分数(area)
  - [-1, 4, A2C2f, [512, True, 4]]

  - [-1, 1, Conv,  [1024, 3, 2]] # 7-P5/32

  # P5/32: 下采样32倍后的特征图
  - [-1, 4, A2C2f, [1024, True, 1]] # 8

# YOLO12-turbo head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4

  # 整合高层语义(P5)和底层细节(P4), 优化中等尺度目标的检测
  - [-1, 2, A2C2f, [512, False, -1]] # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3

  # P3特征包含更多细节信息, 区域注意力帮助定位微小目标(如人脸表情、工业缺陷)
  - [-1, 2, A2C2f, [256, False, -1]] # 14

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]] # cat head P4

  # 对第14层输出与第11层输出的拼接后的输出做注意力调整
  - [-1, 2, A2C2f, [512, False, -1]] # 17

# - ...

  - [[14, 17, 20], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

### A2C2f 实现

一些核心的init参数:
- ​**c1**: 输入通道数（如Backbone中P4层的256维）
- **c2**: 输出通道数（如512或1024），决定特征图的维度
- ​**n**: 堆叠的ABlock或C3k模块数量（默认为1）
- ​**area**: 区域划分数（如4表示水平/垂直4分区），控制注意力计算范围

```python
class A2C2f(nn.Module): 
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        # 用来降低维度, 减少计算量
        c_ = int(c2 * e)  # hidden channels

        # 多注意头数目, 与llm中一致
        num_heads = c_ // 32

        # 1×1卷积将输入从c1降至c_, 减少后续计算量
        self.cv1 = Conv(c1, c_, 1, 1)

        # 拼接n个ABlock输出后，1×1卷积恢复至目标维度c2
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        init_values = 0.01
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        # 初始化 n 个ABBlock块
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )
```

对于某一个 ABBlock 块的定义:

```python
class ABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
```

从上可以看出, 与llm中的定义类似, 但是这里yolo做了一个优化, 便是他提出了area区域注意力:
- 如果area=4, **则会将输入特征图(都是基于特征图做注意力计算)分为4块, 在每一个块里面做attention计算, 相当于每个区域独立不产生关系**, 减少计算量
- 实际上, 对于小目标检测来说, 这个分区域甚至可以调整更大, 以减少计算量, 因为很多小目标与其他远距离区域关系不大
- 具体计算逻辑与llm中基本一致

```python
class AAttn(nn.Module):
    def forward(self, x):
        # 输入x的形状: [Batch, Channels, Height, Width]
        B, C, H, W = x.shape
        N = H * W

        # 通过qk卷积生成Q+K, 展平空间维度并转置为[B, N, 2C]
        # qk卷积同时生成查询（Q）和键（K），通过拼接减少计算量
        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)   # 通过v卷积生成V, 形状保持[B, C, H, W]
        pp = self.pe(v) # 位置编码(如sin-cos或可学习编码), 形状与v相同
        v = v.flatten(2).transpose(1, 2)

        # 按区域划分QK: [B*area, N/area, 2C]
        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        # ...
        else:
            # 计算注意力
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)
```


**如上 维度变化\参数量\运算量的减少 待运行验证**

## todo...


## 引用
1. [source code](https://github.com/sunsmarterjie/yolov12)
2. [ultralytics教程](https://docs.ultralytics.com/zh/models/yolo12/)