# Dynamic Neural Architecture (DNA)

> **动态神经网络 | 纯局部学习 | 无反向传播 | 去中心化 AI**

DNA 是一个基于生物神经科学原理构建的神经网络框架。它的核心设计理念——**每个细胞完全自治**，只依赖局部信息学习，不依赖全局反向传播。

---

## 实验结果总结

经过 16 个版本、50+ 次迭代的系统性实验后，核心结论：

| 版本 | 架构 | 最高准确率 | 关键发现 |
|------|------|-----------|---------|
| v1–v3 | BP + 能量/Vitality | ~34% | 超参数难调，结构可塑性无意义 |
| **v4** | **单像素局部 Hebbian** | **70.74%** | 单像素逻辑回归天花板 |
| v5–v11 | 各种失败尝试 | ~69% | Dopamine/STDP/Extinction 问题 |
| v12 | 2×2 Patch 分区 + Hebbian | 69.53% | Patch 分区稀释信息 |
| **v13** | **全图连接 Hebbian + Extinction** | **91.70%** 🏆 | **批量训练最优，突破 70%** |
| v14 | 非线性细胞 (gain+bias) + WTA | 90.87% | 梯度更新 gain/bias 没调好 |
| **v15** | **Hebbian 置信度规则 + 稳态 bias + 退火 WTA** | **91.76%** | **纯局部规则管理 gain/bias，无 BP** |
| v16 on | **在线学习（推演+学习同时进行）** | **88.82%** 🆕 | **连续学习，从不停止适应** |

### 批量训练最佳结果

| 实验 | 准确率 | 说明 |
|------|-------|------|
| v4 批量 Hebbian (15 epoch) | 70.74% | 单像素逻辑回归天花板 |
| v13 批量 Hebbian + Extinction | **91.70%** | 全图连接 30 细胞，无非线性，无 WTA |
| v15 批量 Hebbian + 非线性 + WTA | **91.76%** 🏆 | Hebbian 置信度 + 稳态 + 退火 WTA |

### 在线学习最佳结果（推演+学习同时进行）

| 实验 | 配置 | best_test | 特点 |
|------|------|----------|------|
| 单 pass 纯在线 | batch=128, lr=0.005 | 76.74% | 一次性看完全部数据 |
| + replay buffer | replay_ratio=0.5, buf=20000 | 84.46% | 混入旧样本防止遗忘 |
| + topk extinction | keep=12, interval=5000 | 87.09% | 有效杀细胞，但 peak 递减 |
| + topk extinction 延长 | keep=12, interval=10000 | 88.80% | 首次 peak 最高，仍递减 |
| **+ soft_energy extinction** 🏆 | drain 低重要性细胞 energy | **88.82%** | **稳定，3 次 extinction 后不衰退** |

**Soft Energy Extinction 是找到的最佳在线学习 extinction 策略**——不硬杀细胞，只 drain 不重要细胞的能量。这保留了权重知识，使每次 extinction 后代模型准确率几乎不下降。

---

## 架构 (v16 最新版)

```
每个细胞:  output = sigmoid(gain * (w · x + bias))

  输入 (784 个像素)
    └── 每个输出类分配 N 个细胞 (初始 8, 最大 32)
        └── 每个细胞有 784 维权重 + 可学习的 gain 和 bias
            ├── gain: Hebbian 置信度规则 (准确率高→变陡峭)
            ├── bias: 稳态规则 (维持 50% 激活率)
            └── WTA 竞争: 退火从全体→top-1 激活
                └── 每个类的总输出 = 获胜细胞的输出均值

学习规则 (纯局部，无 BP):
  Δw     = lr × pixel × error × gain × sigmoid'
  Δgain  = gain_lr × (准确率 - 0.5)   ← 置信度调节
  Δbias  = bias_lr × (设定值 - 激活率) ← 稳态调节

结构可塑性:
  能量 = f(输出与目标的接近程度) - base_cost
  能量 > 150 → 细胞分裂 (继承 50% 能量 + 权重扰动)
  能量 ≤ 0 → 细胞死亡 (grace period 保护新生细胞)
  Extinction (topk 或 soft_energy): 定期淘汰低重要性细胞

在线学习:
  无 train/test 分割，每个样本 forward 后立即学习
  50% replay buffer 防止遗忘
  single pass，模型持续适应
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

### 配置
编辑 `trainer.py` 或 `online.py` 中的 `args` 字典：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `cells_per_class` | 8 | 每类初始细胞数 |
| `max_cells_per_class` | 32 | 每类最大细胞数 |
| `wta_k` | 1 | WTA top-k（1=硬竞争） |
| `wta_anneal_start` | 150 | WTA 退火开始 step |
| `wta_anneal_end` | 500 | WTA 退火结束 step |
| `extinction_mode` | 'soft_energy' | 'median' / 'topk' / 'soft_energy' |
| `extinction_keep` | 12 | topk 模式下保留细胞数 |
| `extinction_interval` | 5000 | extinction 间隔 steps |
| `energy_cost` | 0.5 | 能量消耗速率 |
| `gain_lr` | 0.002 | gain 学习率（置信度规则） |
| `bias_lr` | 0.002 | bias 学习率（稳态规则） |
| `replay_ratio` | 0.5 | 在线学习 replay 比例 |
| `replay_buffer_size` | 20000 | replay 缓冲区大小 |

---

## 项目历程

这个项目起源于对"可动态增删细胞的神经网络"的好奇。经历数十次实验后，最重要的教训是：

1. **生物合理性 ≠ 随机尝试**。生物学经过了亿万年的进化才找到 STDP、多巴胺等机制，简单模仿其表面形式而不理解其功能，必然失败。

2. **Patch 分区是陷阱**。将输入空间切割成独立 patch 并分配独立细胞组，看似"增大感受野"，实际上稀释了每个类的表决权，不如让每个细胞看到全局。

3. **纯局部学习可以达到 90%+**。全图连接的 Hebbian 规则 + 结构可塑性可以逼近逻辑回归集成性能。

4. **结构可塑性是调节器，不是驱动者**。没有有效的学习规则，结构可塑性毫无意义。Soft Energy Extinction 是最好的在线 extinction 方案——保留知识、不衰退。

5. **推演和学习可以同时进行**。DNA 的局部学习规则天然支持在线学习。88.82% 的结果验证了：每个样本只看一次，边推理边学习，细胞在线分裂，是一个可行的学习范式。

---

## 实验数据导出

所有实验日志保存在 `/tmp/online_*.log`：
- `online_topk8.log` — topk extinction, keep=8, interval=2000
- `online_extinct10k.log` — topk extinction, keep=12, interval=10000
- `online_soft.log` — soft_energy extinction, interval=5000

---

## License

MIT
