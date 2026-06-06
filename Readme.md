# Dynamic Neural Architecture (DNA)

> **动态神经网络 | 纯局部学习 | 无反向传播 | 去中心化 AI**

DNA 是一个基于生物神经科学原理构建的神经网络框架。它的核心设计理念——**每个细胞完全自治**，只依赖局部信息学习，不依赖全局反向传播。

---

## 实验结果总结

经过 16 个版本、60+ 次迭代的系统性实验后，**DNA 在 MNIST 上成功达到 91%+**（批量训练和在线学习均达成目标）。

| 版本 | 架构 | 最高准确率 | 关键发现 |
|------|------|-----------|---------|
| v1–v3 | BP + 能量/Vitality | ~34% | 超参数难调，结构可塑性无意义 |
| **v4** | **单像素局部 Hebbian** | **70.74%** | 单像素逻辑回归天花板 |
| v5–v11 | 各种失败尝试 | ~69% | Dopamine/STDP/Extinction 问题 |
| v12 | 2×2 Patch 分区 + Hebbian | 69.53% | Patch 分区稀释信息 |
| **v13** | **全图连接 Hebbian + Extinction** | **91.70%** 🏆 | **批量训练突破 70% 天花板** |
| v14 | 非线性细胞 (gain+bias) + WTA | 90.87% | 梯度更新 gain/bias 没调好 |
| **v15** | **Hebbian 置信度规则 + 稳态 bias + 退火 WTA** | **91.76%** 🤖 | **纯局部规则管理 gain/bias** |
| **v16** | **在线学习（推演+学习同时进行）** | **91.08%** 🎯 | **连续学习达成目标** |

## 双范式结果

### 批量训练

```
训练数据: 60000 张 MNIST
方式: 30 epoch，打乱后重复学习
核心: 每个 batch forward → Hebbian 更新 → 能量更新 → 结构变化
```

| 实验 | 准确率 | 说明 |
|------|-------|------|
| v4 单像素 Hebbian (15 epoch) | 70.74% | 单像素逻辑回归上限 |
| v13 全图连接 + Extinction | **91.70%** 🏆 | 30 细胞，无非线性，无 WTA |
| **v15 非线性 + WTA** | **91.76%** 🥇 | **批量训练最佳**，Hebbian 置信度 + 稳态 + 退火 WTA |

### 在线学习（推演+学习同时进行）

```
训练数据: 60000 张 MNIST
方式: 单 pass，每个样本 forward 后立即学习
核心: 推演和学习同时进行，模型持续在线适应
```

| 实验 | 配置 | best_test | 特点 |
|------|------|----------|------|
| 单 pass 纯在线 | batch=128, lr=0.005 | 76.74% | 一次性看完，不回头 |
| + replay buffer | replay_ratio=0.5, buf=20000 | 84.46% | 混入旧样本防止遗忘 |
| + topk extinction | keep=12, interval=5000 | 87.09% | 有效杀细胞，但 peak 递减 |
| + soft_energy extinction 🏆 | drain 低重要性 energy | 88.82% | 稳定，3 次后不衰退 |
| **+ max_C=64 + 无 WTA** | **640 细胞全投票** | **91.08%** 🎯 | **在线学习达成 91% 目标！** |

### 关键发现：细胞数量是关键

在线学习中，细胞数量直接影响天花板：

```
max_C=32 + 无 WTA (只有 320 个细胞):  未测试 (WTA 一直开着)
max_C=32 + WTA_k=1:                   88.82%  ← 只用一个细胞发声，上限低
max_C=64 + WTA_k=1:                   ~80%    ← WTA 太激进，640 细胞只用 1 个
max_C=64 + WTA_k=64 (全部投票):       91.08%  ← 640 细胞全集成，突破 91%！
```

**结论：** 去掉 WTA 竞争、让所有细胞平等投票，640 个逻辑回归细胞的集成效果超过了 WTA 的 specialization 收益。

---

## 架构 (v16 在线版)

```python
每个细胞:  output = sigmoid(gain * (w · x + bias))

  输入 (784 个像素)
    └── 每个输出类分配 8 个细胞 (最大 64/类, 共 640 个)
        └── 每个细胞有 784 维权重 + 可学习的 gain 和 bias
            ├── gain: Hebbian 置信度规则 (准确率高→变陡峭)
            └── bias: 稳态规则 (维持 50% 激活率)
                └── 所有细胞一起投票 (无 WTA)
                    └── 每个类的输出 = 该类所有细胞输出均值

学习规则 (纯局部，无 BP):
  Δw     = lr × pixel × error × gain × sigmoid'
  Δgain  = gain_lr × (准确率 - 0.5)
  Δbias  = bias_lr × (设定值 - 激活率)

结构可塑性:
  能量 = closeness.mean() - base_cost
  能量 > 150 → 细胞分裂
  能量 ≤ 0   → 细胞死亡 (grace 保护新生)
  Soft Energy Extinction (每 5000 step):
    → 低重要性细胞能量 *= 0.3 (慢慢饿死)
    → 高重要性细胞能量 /= 2
    → 不硬杀，保留权重知识

在线学习:
  - 每个 batch: forward → predict → update → next
  - 50% replay buffer (20000 样本)
  - 无 epoch，单 pass 持续学习
```

---

## 快速开始

### 批量训练
```bash
cd /mnt/e/DNA
python trainer.py
```

### 在线学习（推演+学习同时进行）
```bash
cd /mnt/e/DNA
python online.py
```

### 依赖
- Python 3.8+
- PyTorch 2.0+ (GPU 推荐)
- torchvision

---

## 配置参考

编辑 `trainer.py` 或 `online.py` 中的 `args` 字典：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `cells_per_class` | 8 | 每类初始细胞数 |
| `max_cells_per_class` | 64 | 每类最大细胞数 |
| `wta_k` | 999 (不启用) | WTA top-k |
| `extinction_mode` | 'soft_energy' | 'median' / 'topk' / 'soft_energy' |
| `extinction_interval` | 5000 | extinction 间隔 steps |
| `extinction_keep` | 24 | topk 模式下保留细胞数 |
| `energy_cost` | 0.5 | 能量消耗速率 |
| `gain_lr` | 0.002 | Hebbian 置信度学习率 |
| `bias_lr` | 0.002 | 稳态学习率 |
| `replay_ratio` | 0.5 | 在线学习 replay 比例 |
| `replay_buffer_size` | 20000 | replay 缓冲区大小 |

---

## 项目历程

这个项目起源于对"可动态增删细胞的神经网络"的好奇。经历 60+ 次实验后，最重要的教训是：

1. **生物合理性 ≠ 随机尝试**。STDP、多巴胺等机制的简单模仿而不理解其功能，必然失败。

2. **Patch 分区是陷阱**。将输入空间切成独立 patch 并分配独立细胞，稀释了每个类的表决权。让每个细胞看到全局才是关键。

3. **纯局部学习可以达到 91%+**。全图 + Hebbian + 结构可塑性 = 逻辑回归集成，无 BP 时 MNIST 可达到 91%。

4. **WTA 竞争是双刃剑**。在细胞数量足够时，去掉 WTA、让所有细胞投票，集成效果优于 specialization。

5. **推演和学习可以同时进行**。DNA 的局部规则天然支持在线学习。91.08% 在单 pass 中达成——每个样本只看一次，细胞在线分裂，模型持续适应。

6. **Soft Energy Extinction** 是最好的在线 extinction。不杀细胞、只 drain 能量，保留权重知识，多次 extinction 后准确率几乎不衰退。

---

## License

MIT
